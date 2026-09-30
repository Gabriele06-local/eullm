//! Pulling a model from HuggingFace, once, for both callers.
//!
//! The CLI (`eullm pull`) and the API (`POST /api/pull`) need exactly the same
//! sequence — resolve the repo, fetch every shard, fetch the projector beside
//! them, write the manifest — and differ only in how they report it: one
//! prints, the other streams NDJSON. So the sequence lives here and the
//! reporting is a channel.
//!
//! Written this way on purpose rather than implemented twice. Two copies of a
//! prompt builder is how the multimodal path spent a month sending Gemma turn
//! markers to every model, because the fix landed on one copy; two copies of a
//! download path would go the same way the first time a repo layout changes.

use std::path::Path;

use tokio::sync::mpsc;

use crate::models::store::{HfFile, ModelStore};
use crate::registry::{self, HfRef};

/// Derive a filesystem-safe model id from a HuggingFace ref. Uses the repo
/// name (last path segment), lowercased and sanitized like `url_to_model_id`,
/// with the quant appended when one was requested so different quants of the
/// same repo coexist:  `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M` → `qwen3-8b-gguf-q4_k_m`.
pub fn hf_ref_to_model_id(hf: &registry::HfRef) -> String {
    let repo_name = hf.repo.rsplit('/').next().unwrap_or(&hf.repo);
    let base = match hf.quant.as_deref() {
        Some(q) => format!("{repo_name}-{q}"),
        None => repo_name.to_string(),
    };
    let id: String = base
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    let id = id.trim_matches('-').to_string();
    if id.is_empty() {
        "model".to_string()
    } else {
        id
    }
}


/// What `reuse_stored_weights` found: the file a pull is about to download,
/// already on disk under another id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlreadyStored {
    /// `from` holds the file, and the requested id now shares it through hard
    /// links. The caller still writes the requested id's manifest.
    Linked { from: String },
    /// `from` holds the file but it could not be linked (`reason`). Nothing
    /// was created for the requested id, and nothing is to be downloaded:
    /// the model is `from`, and that is the id to run.
    Unlinked { from: String, reason: String },
}

/// Before downloading `wanted` as model `id`: whether another stored model
/// already has it, and if so, link it in as `id` instead.
///
/// A catalog id and an `hf.co` ref are two names for one download when they
/// resolve to the same file of the same repository — `qwen3-8b` and
/// `hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M` are both `Qwen3-8B-Q4_K_M.gguf` in
/// `unsloth/Qwen3-8B-GGUF`. Each pull used to check only its own id, so the
/// second one downloaded the same 5 GB into a second directory. The same
/// quantization from another repository is not the same file (someone else
/// quantized it), and another quantization of the same repository is a
/// different file: neither is reused.
///
/// `files` is every file the weights consist of, as stored — all shards of a
/// split. A stored model counts only when all of them are there; the first
/// shard alone would link in a model that cannot load. `None`: no stored
/// model has it, download as usual.
pub fn reuse_stored_weights(
    store: &ModelStore,
    wanted: &HfFile,
    id: &str,
    files: &[String],
) -> Option<AlreadyStored> {
    reuse_stored_weights_with(store, wanted, id, files, &|src, dst| {
        std::fs::hard_link(src, dst)
    })
}

fn reuse_stored_weights_with(
    store: &ModelStore,
    wanted: &HfFile,
    id: &str,
    files: &[String],
    link: &dyn Fn(&Path, &Path) -> std::io::Result<()>,
) -> Option<AlreadyStored> {
    // Sorted, so which copy is picked when there are several does not depend
    // on directory iteration order.
    let mut holders: Vec<String> = store
        .list()
        .ok()?
        .into_iter()
        .filter(|m| m.id != id && m.holds(wanted))
        .map(|m| m.id)
        .filter(|holder| {
            let dir = store.model_path(holder);
            files.iter().all(|f| dir.join(f).is_file())
        })
        .collect();
    holders.sort();
    let from = holders.into_iter().next()?;
    Some(match store.link_files(&from, id, files, link) {
        Ok(()) => AlreadyStored::Linked { from },
        Err(e) => AlreadyStored::Unlinked {
            from,
            reason: e.to_string(),
        },
    })
}

/// After `reuse_stored_weights` linked `from`'s weights into `id`: link
/// `from`'s projector as well when it is the one `id` wants. Returns its file
/// name when it was linked; `None` leaves the projector to be downloaded the
/// usual way, which is also what happens when `from` has a different one.
pub fn link_stored_projector(
    store: &ModelStore,
    from: &str,
    id: &str,
    wanted: &HfFile,
) -> Option<String> {
    let holder = store.get(from).ok().flatten()?;
    if !holder.projector_is(wanted) {
        return None;
    }
    let name = wanted.file_name().to_string();
    store
        .link_files(from, id, std::slice::from_ref(&name), &|src, dst| {
            std::fs::hard_link(src, dst)
        })
        .ok()?;
    Some(name)
}

/// What a pull reports as it goes.
#[derive(Debug, Clone)]
pub enum PullEvent {
    /// A step began. Human-readable, and the `status` field Ollama clients
    /// display.
    Status(String),
    /// Bytes moved for one file. `total` is 0 when the server sent no
    /// Content-Length, which is the same convention the download layer uses.
    Progress {
        file: String,
        completed: u64,
        total: u64,
    },
    /// The model is on disk and usable, stored under this id.
    Done { id: String },
    /// The pull failed and nothing was left behind.
    Failed(String),
}

/// Send and ignore a closed receiver: a client that hangs up mid-download
/// should not turn into an error inside the download.
fn emit(tx: &mpsc::Sender<PullEvent>, ev: PullEvent) {
    let _ = tx.try_send(ev);
}

/// Fetch `hf` into `store`, reporting through `tx`.
///
/// Returns the stored model id on success. That is `id`, except when the
/// same file was already on disk under another id and could not be linked
/// (see `reuse_stored_weights`): nothing is downloaded then, and the id
/// returned — and sent in `Done` — is the one that has it. Every failure path
/// removes the model directory first: a partial split cannot be loaded, and
/// half a model that `eullm list` shows as present is worse than no model at
/// all.
pub async fn pull_from_huggingface(
    store: &ModelStore,
    hf: &HfRef,
    id: &str,
    tx: mpsc::Sender<PullEvent>,
) -> Result<String, String> {
    emit(
        &tx,
        PullEvent::Status(format!("resolving {} on HuggingFace", hf.repo)),
    );
    let filenames = registry::resolve_hf_gguf(hf)
        .await
        .map_err(|e| format!("could not resolve a GGUF to download: {e}"))?;
    // The whole split is named by its first shard, as in the manifest.
    let weights = HfFile::new(&hf.repo, &filenames[0]);

    let model_dir = store.model_path(id);
    // Each name is the repo-relative path and can carry a subdirectory when
    // the repo groups quantizations (`UD-Q4_K_XL/Model-…-00001-of-00004.gguf`).
    // The remote path is what the download needs; locally the model already
    // has its own directory, so that prefix is redundant and the file is
    // stored under its bare name. Shards keep their `-NNNNN-of-TOTAL` suffix:
    // only the directory prefix is dropped.
    let leaves: Vec<String> = filenames
        .iter()
        .map(|f| f.rsplit('/').next().unwrap_or(f).to_string())
        .collect();
    // The manifest names the first shard. llama.cpp reads the split count from
    // its header and opens the siblings itself, which is why they all have to
    // land in the same directory.
    let leaf = leaves[0].clone();

    // Only now, with the file known: what the ref resolves to is what decides
    // whether it is already here, and the repo listing is what resolves it.
    let linked_from = match reuse_stored_weights(store, &weights, id, &leaves) {
        Some(AlreadyStored::Linked { from }) => {
            emit(
                &tx,
                PullEvent::Status(format!(
                    "{leaf} is already downloaded as {from}: linked it as {id} \
                     instead of downloading it again (no extra disk space)"
                )),
            );
            Some(from)
        }
        Some(AlreadyStored::Unlinked { from, reason }) => {
            emit(
                &tx,
                PullEvent::Status(format!(
                    "{leaf} is already downloaded as {from}, and could not be linked \
                     as {id} ({reason}). Nothing was downloaded: use {from}"
                )),
            );
            emit(&tx, PullEvent::Done { id: from.clone() });
            return Ok(from);
        }
        None => None,
    };

    if linked_from.is_none() {
        if filenames.len() > 1 {
            emit(
                &tx,
                PullEvent::Status(format!("pulling {} in {} shards", hf.repo, filenames.len())),
            );
        }

        for (remote, local) in filenames.iter().zip(leaves.iter()) {
            emit(&tx, PullEvent::Status(format!("pulling {local}")));
            let progress = file_progress(&tx, local);
            if let Err(e) = registry::download_from_huggingface(
                &hf.repo,
                remote,
                &model_dir.join(local),
                None,
                Some(progress),
            )
            .await
            {
                let _ = store.delete(id);
                return Err(format!("download failed on {local}: {e}"));
            }
        }
    }

    // A vision repo ships the projector beside the weights, and without it the
    // model loads but cannot see. llama.cpp's own `-hf` fetches both, and a
    // user who has to notice the second file and pass `--mmproj` by hand is
    // being asked to know something the repo layout already says.
    let mmproj_name = match registry::list_hf_ggufs(&hf.repo).await {
        Ok(files) => files.into_iter().find(|f| registry::is_mmproj(f)),
        Err(_) => None,
    };
    let mut mmproj_stored: Option<String> = None;
    if let Some(name) = mmproj_name {
        let projector = name.rsplit('/').next().unwrap_or(&name).to_string();
        // Weights linked from another id bring that id's projector along when
        // it is this one. Otherwise it is fetched like on any other pull.
        mmproj_stored = linked_from
            .as_deref()
            .and_then(|from| link_stored_projector(store, from, id, &HfFile::new(&hf.repo, &name)));
        if mmproj_stored.is_none() {
            emit(
                &tx,
                PullEvent::Status(format!("pulling projector {projector}")),
            );
            let progress = file_progress(&tx, &projector);
            match registry::download_from_huggingface(
                &hf.repo,
                &name,
                &model_dir.join(&projector),
                None,
                Some(progress),
            )
            .await
            {
                Ok(()) => mmproj_stored = Some(projector),
                // The weights are already on disk and usable for text. Losing
                // the projector costs image and audio input, not the model, so
                // it is reported and the pull still succeeds.
                Err(e) => emit(
                    &tx,
                    PullEvent::Status(format!(
                        "projector download failed ({e}); text still works, \
                         re-run the pull or pass --mmproj"
                    )),
                ),
            }
        }
    }

    // Every shard, not just the first: the recorded size is what the model
    // costs on disk, and showing 30 GB for a 111 GB split would be worse than
    // showing nothing.
    let size: u64 = leaves
        .iter()
        .filter_map(|l| std::fs::metadata(model_dir.join(l)).ok())
        .map(|m| m.len())
        .sum();

    store
        .write_external_manifest(
            id,
            &leaf,
            &hf.original,
            size,
            mmproj_stored.as_deref(),
            Some(&weights),
        )
        .map_err(|e| format!("the weights are in place but the manifest write failed: {e}"))?;

    emit(&tx, PullEvent::Done { id: id.to_string() });
    Ok(id.to_string())
}

/// A progress callback that forwards into the event channel.
///
/// `try_send` on a bounded channel, so a slow or vanished consumer drops
/// progress ticks instead of stalling the download. Losing a tick costs a
/// smoother bar; blocking a download on a browser that stopped reading costs
/// the download.
fn file_progress(tx: &mpsc::Sender<PullEvent>, file: &str) -> registry::ProgressCallback {
    let tx = tx.clone();
    let file = file.to_string();
    Box::new(move |completed, total| {
        let _ = tx.try_send(PullEvent::Progress {
            file: file.clone(),
            completed,
            total,
        });
    })
}

#[cfg(test)]
mod reuse_tests {
    use super::*;
    use crate::models::catalog::{self, CatalogEntry};
    use std::fs;
    use std::path::PathBuf;

    /// A store in a directory of its own, removed afterwards.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let d = std::env::temp_dir().join(format!("eullm-reuse-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&d).expect("scratch dir");
            Self(d)
        }
        fn store(&self) -> ModelStore {
            ModelStore::at(self.0.clone())
        }
        /// A model directory as a pull leaves it: the files, then the
        /// manifest.
        fn model(&self, id: &str, manifest: serde_json::Value, files: &[&str]) {
            let dir = self.0.join(id);
            fs::create_dir_all(&dir).expect("model dir");
            for f in files {
                fs::write(dir.join(f), format!("weights of {f}")).expect("weights");
            }
            fs::write(dir.join("manifest.json"), manifest.to_string()).expect("manifest");
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn qwen3_8b() -> &'static CatalogEntry {
        catalog::find_model("qwen3-8b").expect("qwen3-8b is in the catalog")
    }

    fn catalog_weights(entry: &CatalogEntry) -> HfFile {
        HfFile::new(&entry.hf_repo, &entry.hf_filename)
    }

    /// The manifest 0.7.20 wrote for a catalog pull: no `hf_repo`, and the
    /// catalog's digest of the file the download was verified against.
    fn legacy_catalog_manifest(entry: &CatalogEntry, digest: &str) -> serde_json::Value {
        serde_json::json!({
            "id": entry.id, "name": entry.name, "description": entry.description,
            "languages": entry.languages, "base": entry.base(), "vram_gb": entry.vram_gb,
            "size_bytes": entry.size_bytes, "license": entry.license, "digest": digest,
            "pulled_at": "2026-09-01T00:00:00Z", "status": "ready",
            "gguf_file": entry.hf_filename,
        })
    }

    /// The manifest 0.7.20 wrote for `eullm pull hf.co/...`: the ref in the
    /// description and only the stored file name.
    fn legacy_hf_manifest(id: &str, hf_ref: &str, gguf_file: &str) -> serde_json::Value {
        serde_json::json!({
            "id": id, "name": id, "description": format!("External model pulled from {hf_ref}"),
            "languages": [], "base": id, "vram_gb": 0, "size_bytes": 20, "license": "unknown",
            "digest": "", "pulled_at": "2026-09-01T00:00:00Z", "status": "ready",
            "gguf_file": gguf_file,
        })
    }

    fn read(path: PathBuf) -> String {
        fs::read_to_string(path).expect("readable weights")
    }

    #[cfg(unix)]
    fn same_inode(a: &std::path::Path, b: &std::path::Path) -> bool {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(a).unwrap().ino() == fs::metadata(b).unwrap().ino()
    }

    // `cmd_pull` asks `gguf_path` about the id itself first and stops there
    // when it is present, as it always did. Reuse must never take the id
    // itself for "another id that has it", nor create anything for it.
    #[test]
    fn the_id_that_already_has_the_file_is_left_as_it_is() {
        let s = Scratch::new();
        let entry = qwen3_8b();
        s.model(
            &entry.id,
            legacy_catalog_manifest(entry, &entry.digest),
            &[&entry.hf_filename],
        );
        let store = s.store();

        assert!(store.gguf_path(&entry.id).is_some());
        let files = [entry.hf_filename.clone()];
        assert_eq!(
            reuse_stored_weights(&store, &catalog_weights(entry), &entry.id, &files),
            None
        );
        assert_eq!(store.list().expect("list").len(), 1);
    }

    // The report: `eullm pull hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M`, then
    // `eullm pull qwen3-8b`, downloaded the same file twice. With the first
    // pull's manifest exactly as 0.7.20 wrote it.
    #[test]
    fn pulling_the_catalog_id_links_what_a_hub_ref_pull_stored() {
        let s = Scratch::new();
        let entry = qwen3_8b();
        let hub_id = "qwen3-8b-gguf-q4_k_m";
        s.model(
            hub_id,
            legacy_hf_manifest(
                hub_id,
                "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M",
                &entry.hf_filename,
            ),
            &[&entry.hf_filename],
        );
        let store = s.store();
        let weights = catalog_weights(entry);

        let files = [entry.hf_filename.clone()];
        assert_eq!(
            reuse_stored_weights(&store, &weights, &entry.id, &files),
            Some(AlreadyStored::Linked {
                from: hub_id.to_string()
            })
        );
        // What `cmd_pull` writes next.
        store
            .write_manifest(
                entry,
                "ready",
                Some(&entry.hf_filename),
                None,
                Some(&weights),
            )
            .expect("manifest");

        let linked = store.gguf_path(&entry.id).expect("the catalog id runs");
        let original = store.gguf_path(hub_id).expect("the hub id still runs");
        assert_eq!(read(linked.clone()), read(original.clone()));
        #[cfg(unix)]
        assert!(same_inode(&linked, &original), "a copy, not a link");

        let manifest = store.get(&entry.id).expect("get").expect("manifest");
        assert_eq!(manifest.hf_repo.as_deref(), Some(entry.hf_repo.as_str()));
        assert_eq!(
            manifest.hf_filename.as_deref(),
            Some(entry.hf_filename.as_str())
        );
    }

    // The other order, and a ref typed in another case: the Hub resolves
    // repository names without regard to case, so this is still the file the
    // catalog pull stored.
    #[test]
    fn pulling_a_hub_ref_links_what_the_catalog_pull_stored() {
        let s = Scratch::new();
        let entry = qwen3_8b();
        s.model(
            &entry.id,
            legacy_catalog_manifest(entry, &entry.digest),
            &[&entry.hf_filename],
        );
        let store = s.store();
        let hub_id = "qwen3-8b-gguf-q4_k_m";
        let wanted = HfFile::new("UNSLOTH/qwen3-8b-gguf", &entry.hf_filename);

        let files = [entry.hf_filename.clone()];
        assert_eq!(
            reuse_stored_weights(&store, &wanted, hub_id, &files),
            Some(AlreadyStored::Linked {
                from: entry.id.clone()
            })
        );
        // What `pull_from_huggingface` writes next.
        store
            .write_external_manifest(
                hub_id,
                &entry.hf_filename,
                "hf.co/UNSLOTH/qwen3-8b-gguf:Q4_K_M",
                20,
                None,
                Some(&wanted),
            )
            .expect("manifest");

        assert!(store.gguf_path(hub_id).is_some(), "the hub id runs");
        assert!(
            store.gguf_path(&entry.id).is_some(),
            "the catalog id still runs"
        );
        // A manifest that records the file is itself recognised from then on.
        let recorded = store.get(hub_id).expect("get").expect("manifest");
        assert!(recorded.holds(&catalog_weights(entry)));
    }

    #[test]
    fn another_quantization_of_the_same_repo_is_not_the_same_file() {
        let s = Scratch::new();
        let entry = qwen3_8b();
        let hub_id = "qwen3-8b-gguf-q4_k_m";
        s.model(
            hub_id,
            legacy_hf_manifest(
                hub_id,
                "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M",
                &entry.hf_filename,
            ),
            &[&entry.hf_filename],
        );
        let store = s.store();

        let q8 = HfFile::new(&entry.hf_repo, "Qwen3-8B-Q8_0.gguf");
        let files = ["Qwen3-8B-Q8_0.gguf".to_string()];
        assert_eq!(
            reuse_stored_weights(&store, &q8, "qwen3-8b-gguf-q8_0", &files),
            None
        );
        assert!(!store.model_path("qwen3-8b-gguf-q8_0").exists());
    }

    // `Qwen/Qwen3-8B-GGUF` publishes a file with the very same name as the
    // one in `unsloth/Qwen3-8B-GGUF`, quantized by someone else: a different
    // file, whatever it is called.
    #[test]
    fn the_same_name_in_another_repository_is_not_the_same_file() {
        let s = Scratch::new();
        let entry = qwen3_8b();
        s.model(
            &entry.id,
            legacy_catalog_manifest(entry, &entry.digest),
            &[&entry.hf_filename],
        );
        let store = s.store();

        let other = HfFile::new("Qwen/Qwen3-8B-GGUF", &entry.hf_filename);
        let files = [entry.hf_filename.clone()];
        assert_eq!(
            reuse_stored_weights(&store, &other, "qwen3-8b-gguf-q4_k_m", &files),
            None
        );
    }

    // A catalog manifest names a file through the catalog only while the
    // catalog still carries the digest it was verified against. Once the
    // entry moves to another file, the old download is not that file.
    #[test]
    fn a_catalog_pull_from_an_older_catalog_is_not_taken_for_todays_file() {
        let s = Scratch::new();
        let entry = qwen3_8b();
        s.model(
            &entry.id,
            legacy_catalog_manifest(entry, "sha256:0000"),
            &[&entry.hf_filename],
        );
        let store = s.store();

        let files = [entry.hf_filename.clone()];
        assert_eq!(
            reuse_stored_weights(
                &store,
                &catalog_weights(entry),
                "qwen3-8b-gguf-q4_k_m",
                &files
            ),
            None
        );
    }

    #[test]
    fn removing_either_id_leaves_the_other_usable() {
        let s = Scratch::new();
        let entry = qwen3_8b();
        let hub_id = "qwen3-8b-gguf-q4_k_m";
        s.model(
            hub_id,
            legacy_hf_manifest(
                hub_id,
                "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M",
                &entry.hf_filename,
            ),
            &[&entry.hf_filename],
        );
        let store = s.store();
        let weights = catalog_weights(entry);
        let files = [entry.hf_filename.clone()];
        let content = read(store.gguf_path(hub_id).unwrap());

        // Link it in as the catalog id, then remove the id it came from.
        assert!(matches!(
            reuse_stored_weights(&store, &weights, &entry.id, &files),
            Some(AlreadyStored::Linked { .. })
        ));
        store
            .write_manifest(
                entry,
                "ready",
                Some(&entry.hf_filename),
                None,
                Some(&weights),
            )
            .expect("manifest");
        let removed = store.delete(hub_id).expect("rm").expect("was there");
        assert!(removed.freed > 0, "its own manifest, at least, is gone");
        #[cfg(unix)]
        assert_eq!(
            removed.shared,
            content.len() as u64,
            "the weights live on under the catalog id and must not count as freed"
        );
        let path = store
            .gguf_path(&entry.id)
            .expect("the catalog id still runs");
        assert_eq!(read(path), content);

        // And the other way round: link it back under a new id and remove
        // that one. The linked manifest is the one recognised now.
        assert!(matches!(
            reuse_stored_weights(&store, &weights, hub_id, &files),
            Some(AlreadyStored::Linked { .. })
        ));
        store.delete(hub_id).expect("rm");
        let path = store
            .gguf_path(&entry.id)
            .expect("the catalog id still runs");
        assert_eq!(read(path), content);
    }

    // A store whose directories cannot share a file — two filesystems, or
    // one without hard links (FAT32, exFAT) — must still not download the
    // same file again: the caller says which id has it instead.
    #[test]
    fn a_link_that_cannot_be_made_downloads_nothing_and_names_the_id_that_has_it() {
        let s = Scratch::new();
        let entry = qwen3_8b();
        let hub_id = "qwen3-8b-gguf-q4_k_m";
        s.model(
            hub_id,
            legacy_hf_manifest(
                hub_id,
                "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M",
                &entry.hf_filename,
            ),
            &[&entry.hf_filename],
        );
        let store = s.store();

        let files = [entry.hf_filename.clone()];
        let outcome = reuse_stored_weights_with(
            &store,
            &catalog_weights(entry),
            &entry.id,
            &files,
            &|_, _| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "hard links are not supported",
                ))
            },
        );
        assert_eq!(
            outcome,
            Some(AlreadyStored::Unlinked {
                from: hub_id.to_string(),
                reason: "hard links are not supported".to_string(),
            })
        );
        assert!(
            !store.model_path(&entry.id).exists(),
            "nothing may be left behind under the id that could not be linked"
        );
        assert!(store.gguf_path(hub_id).is_some());
    }

    // A split is named by its first shard, but the first shard alone does
    // not load. Every shard has to be there to count as the same download.
    #[test]
    fn a_split_is_reused_only_when_every_shard_is_there() {
        let s = Scratch::new();
        let shards = [
            "Model-Q8_0-00001-of-00002.gguf".to_string(),
            "Model-Q8_0-00002-of-00002.gguf".to_string(),
        ];
        let wanted = HfFile::new("someone/Model-GGUF", "Q8_0/Model-Q8_0-00001-of-00002.gguf");
        s.model(
            "model-gguf-q8_0",
            serde_json::json!({
                "id": "model-gguf-q8_0", "name": "model-gguf-q8_0",
                "description": "External model pulled from hf.co/someone/Model-GGUF:Q8_0",
                "languages": [], "base": "model-gguf-q8_0", "vram_gb": 0, "size_bytes": 40,
                "license": "unknown", "digest": "", "pulled_at": "", "status": "ready",
                "gguf_file": shards[0], "hf_repo": wanted.repo, "hf_filename": wanted.path,
            }),
            &[&shards[0], &shards[1]],
        );
        let store = s.store();

        assert!(matches!(
            reuse_stored_weights(&store, &wanted, "model-gguf-q8", &shards),
            Some(AlreadyStored::Linked { .. })
        ));
        for shard in &shards {
            assert!(store.model_path("model-gguf-q8").join(shard).is_file());
        }

        // The linked id has no manifest yet, so the original is the only
        // candidate; without its second shard it is not the model.
        fs::remove_file(store.model_path("model-gguf-q8_0").join(&shards[1])).unwrap();
        assert_eq!(
            reuse_stored_weights(&store, &wanted, "model-q8", &shards),
            None,
            "an incomplete split is not the model"
        );
    }

    // The projector comes along with the weights when it is the one wanted,
    // and only then: another projector is fetched as on any other pull.
    #[test]
    fn the_projector_is_linked_when_it_is_the_same_one() {
        let s = Scratch::new();
        let entry = catalog::find_model("gemma-4-e4b").expect("in the catalog");
        let projector = entry.mmproj_filename.clone().expect("a vision model");
        let mut manifest = legacy_catalog_manifest(entry, &entry.digest);
        manifest["mmproj_file"] = serde_json::json!(projector);
        s.model(&entry.id, manifest, &[&entry.hf_filename, &projector]);
        let store = s.store();

        let hub_id = "gemma-4-e4b-it-gguf-q4_k_m";
        let files = [entry.hf_filename.clone()];
        let from = match reuse_stored_weights(&store, &catalog_weights(entry), hub_id, &files) {
            Some(AlreadyStored::Linked { from }) => from,
            other => panic!("expected a link, got {other:?}"),
        };
        assert_eq!(
            link_stored_projector(
                &store,
                &from,
                hub_id,
                &HfFile::new(&entry.hf_repo, "mmproj-BF16.gguf")
            ),
            None,
            "a different projector is not linked"
        );
        assert_eq!(
            link_stored_projector(
                &store,
                &from,
                hub_id,
                &HfFile::new(&entry.hf_repo, &projector)
            ),
            Some(projector.clone())
        );
        assert!(store.model_path(hub_id).join(&projector).is_file());
    }
}
