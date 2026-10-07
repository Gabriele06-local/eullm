//! `eullm update`: put the latest stable release in place of this one.
//!
//! Nothing here runs unless the command is typed. The engine never looks for
//! a new version by itself: no request leaves the machine that its user did
//! not ask for, and that includes asking github.com what the latest release
//! is.
//!
//! The release downloaded is the same build as the one running — CPU, CUDA,
//! Vulkan, ROCm, for the same system — because each release job bakes the
//! name of the asset it publishes into the binary (`EULLM_RELEASE_ASSET`,
//! see `build.rs`). A build from source has no such name, and is told how to
//! update instead of being replaced by a guess.

use std::path::{Path, PathBuf};
use std::time::Duration;

const REPO: &str = "eullm/eullm";
const TAG_PREFIX: &str = "EuLLM-v";

/// The release asset this binary is published as (`eullm-linux-x64`,
/// `eullm-windows-x64-vulkan.zip`, ...). Empty for a build from source or in
/// a container image.
const RELEASE_ASSET: &str = env!("EULLM_RELEASE_ASSET");

/// What the files of a replaced installation are renamed to, until they can
/// be deleted. On Windows a running exe and the DLLs it loaded cannot be
/// deleted or overwritten, but they can be renamed.
const OLD_SUFFIX: &str = ".old";

/// A version as the release tags carry it. Pre-release tags (`-rc1`, `-beta`)
/// do not parse: `releases/latest` never points at one, and a running
/// pre-release is compared by its numbers alone.
fn parse_version(s: &str) -> Option<(u64, u64, u64)> {
    let s = s.trim().trim_start_matches('v');
    let mut parts = s.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

/// The running version, without a pre-release suffix.
fn running_version() -> Option<(u64, u64, u64)> {
    let v = env!("CARGO_PKG_VERSION");
    parse_version(v.split('-').next().unwrap_or(v))
}

/// `EuLLM-v0.7.40` out of the address `releases/latest` redirects to.
fn tag_from_location(location: &str) -> Option<&str> {
    let tag = location
        .trim_end_matches('/')
        .rsplit_once("/releases/tag/")?
        .1;
    (!tag.is_empty() && !tag.contains('/')).then_some(tag)
}

/// The SHA-256 that `checksums.txt` lists for `asset`. Lines are
/// `<hash>  <name>`; releases up to 0.7.20 wrote `<hash>  <dir>/<name>`, so
/// the name is matched after the last `/`, as the installers do.
fn checksum_for(checksums: &str, asset: &str) -> Option<String> {
    checksums.lines().find_map(|line| {
        let (hash, name) = line.trim().split_once(char::is_whitespace)?;
        let name = name.trim();
        (name.rsplit('/').next() == Some(asset) && hash.len() == 64).then(|| hash.to_string())
    })
}

/// Whether `asset` is a release ZIP in ROCm's own layout, which the update
/// below cannot install: the Windows ROCm build keeps eullm.exe and the DLLs
/// in `bin\`, with kernel folders under it, and its libraries' GPU code in
/// `.kpack\` beside it, while the update replaces the files next to the
/// running exe.
fn keeps_rocm_layout(asset: &str) -> bool {
    asset == "eullm-windows-x64-rocm.zip"
}

/// A client that does not follow redirects, for reading where
/// `releases/latest` points.
fn no_redirect_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(concat!("eullm/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(60))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| format!("could not set up the HTTP client: {e}"))
}

/// The tag of the latest stable release. GitHub answers
/// `releases/latest` with a redirect to the release's own page, whose
/// address ends in its tag; this avoids the REST API and its rate limit.
async fn latest_tag() -> Result<String, String> {
    let url = format!("https://github.com/{REPO}/releases/latest");
    let response = no_redirect_client()?
        .head(&url)
        .send()
        .await
        .map_err(|e| format!("could not reach {url}: {e}"))?;
    let location = response
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            format!(
                "{url} answered {} without saying where the latest release is",
                response.status()
            )
        })?;
    tag_from_location(location)
        .map(str::to_string)
        .ok_or_else(|| format!("{url} pointed at {location}, which names no release"))
}

/// Run `eullm update`, or `eullm update --check` when `check_only`.
pub async fn run(check_only: bool) -> Result<(), String> {
    let current = env!("CARGO_PKG_VERSION");
    println!("This is EuLLM {current}. Asking github.com for the latest release...");
    let tag = latest_tag().await?;
    let latest_str = tag.strip_prefix(TAG_PREFIX).unwrap_or(&tag);
    let latest = parse_version(latest_str)
        .ok_or_else(|| format!("the latest release, {tag}, has a version this cannot read"))?;
    let running = running_version()
        .ok_or_else(|| format!("this binary's own version, {current}, cannot be read"))?;

    if latest <= running {
        println!("EuLLM {current} is up to date (latest release: {latest_str}).");
        return Ok(());
    }
    println!("EuLLM {latest_str} is available.");
    if check_only {
        println!("Run `eullm update` to install it.");
        return Ok(());
    }

    if RELEASE_ASSET.is_empty() {
        return Err(format!(
            "this binary was not built by the release workflow (it was built from source \
             or in a container), so there is no release download it is known to match. \
             Download {latest_str} from https://github.com/{REPO}/releases/latest, or \
             rebuild from the {tag} tag."
        ));
    }
    // Said before a download of a few hundred MB rather than after it.
    if keeps_rocm_layout(RELEASE_ASSET) {
        return Err(format!(
            "{RELEASE_ASSET} keeps ROCm's own layout (eullm.exe in bin\\, its libraries' \
             GPU code in .kpack\\ beside it), which `eullm update` cannot replace yet. \
             Download it from https://github.com/{REPO}/releases/latest and extract it \
             over the folder that holds bin\\."
        ));
    }

    let exe = std::env::current_exe()
        .and_then(|p| p.canonicalize())
        .map_err(|e| format!("could not find where this binary is installed: {e}"))?;
    let shown_exe = display_path(&exe);
    if is_store_install(&exe) {
        return Err(format!(
            "this EuLLM was installed from the Microsoft Store ({shown_exe}), \
             which updates it itself."
        ));
    }
    let dir = exe
        .parent()
        .ok_or_else(|| format!("{shown_exe} has no parent directory"))?
        .to_path_buf();
    let exe_name = exe
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("{shown_exe} has no file name"))?
        .to_string();

    // In the install directory, so every rename below stays on one volume,
    // and created first, so a directory this user cannot write to is found
    // before a download of up to a few hundred MB rather than after it.
    let staging = dir.join(format!(".eullm-update-{latest_str}"));
    if staging.exists() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    std::fs::create_dir_all(&staging).map_err(|e| {
        format!(
            "cannot write to {}: {e}. Run the update as the user that installed EuLLM: \
             for a system-wide install, `sudo eullm update` on Linux or macOS, or a \
             terminal opened as administrator on Windows.",
            display_path(&dir)
        )
    })?;

    let result = install(&tag, &dir, &exe_name, &staging, latest_str).await;
    let _ = std::fs::remove_dir_all(&staging);
    let leftovers = result?;

    println!(
        "Updated to EuLLM {latest_str} ({RELEASE_ASSET}) in {}.",
        display_path(&dir)
    );
    if leftovers > 0 {
        println!(
            "{leftovers} file(s) of the old version are still in use and stay there as *{OLD_SUFFIX}; \
             the next update removes them."
        );
    }
    println!("A running `eullm serve` or `eullm run` keeps the old version until it is restarted.");
    Ok(())
}

/// Download, verify, unpack, test and put in place. Returns how many old
/// files could not be deleted yet.
async fn install(
    tag: &str,
    dir: &Path,
    exe_name: &str,
    staging: &Path,
    version: &str,
) -> Result<usize, String> {
    let base = format!("https://github.com/{REPO}/releases/download/{tag}");

    let checksums = reqwest::Client::builder()
        .user_agent(concat!("eullm/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| format!("could not set up the HTTP client: {e}"))?
        .get(format!("{base}/checksums.txt"))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("could not download {tag}'s checksums.txt: {e}"))?
        .text()
        .await
        .map_err(|e| format!("could not read {tag}'s checksums.txt: {e}"))?;
    let expected = checksum_for(&checksums, RELEASE_ASSET).ok_or_else(|| {
        format!(
            "{tag} does not list {RELEASE_ASSET} in its checksums.txt: that build may have \
             failed for this release. Nothing was changed."
        )
    })?;

    println!("Downloading {RELEASE_ASSET}...");
    let download = staging.join(RELEASE_ASSET);
    let progress: crate::registry::ProgressCallback = {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};
        let last = Arc::new(AtomicU64::new(0));
        Box::new(move |done, total| {
            if done - last.load(Ordering::Relaxed) > 10_000_000 || (total > 0 && done >= total) {
                last.store(done, Ordering::Relaxed);
                eprint!("\r  {}", crate::registry::format_progress(done, total));
                let _ = std::io::Write::flush(&mut std::io::stderr());
            }
        })
    };
    let downloaded = crate::registry::download_file(
        &format!("{base}/{RELEASE_ASSET}"),
        &download,
        Some(&expected),
        Some(progress),
    )
    .await;
    eprintln!();
    downloaded.map_err(|e| format!("download failed: {e}. Nothing was changed."))?;
    println!("Checksum OK.");

    let unpacked = staging.join("files");
    std::fs::create_dir_all(&unpacked)
        .map_err(|e| format!("could not create {}: {e}", display_path(&unpacked)))?;
    let new_exe_name = if RELEASE_ASSET.ends_with(".zip") {
        unzip(&download, &unpacked)?;
        "eullm.exe"
    } else {
        let target = unpacked.join(exe_name);
        std::fs::rename(&download, &target)
            .map_err(|e| format!("could not move the download into place: {e}"))?;
        make_executable(&target)?;
        exe_name
    };

    check_starts(&unpacked.join(new_exe_name), version)?;

    let moves = planned_moves(&unpacked, new_exe_name, exe_name, dir)?;
    remove_old_leftovers(&moves);
    swap(&moves)
}

/// Unpack a release ZIP with the system's own tools: `tar`, which Windows
/// has shipped since 10 1803 and which reads ZIP files, then PowerShell's
/// Expand-Archive. No ZIP library is linked for one command.
fn unzip(zip: &Path, into: &Path) -> Result<(), String> {
    let tar = std::process::Command::new("tar")
        .arg("-xf")
        .arg(zip)
        .arg("-C")
        .arg(into)
        .status();
    if matches!(tar, Ok(s) if s.success()) {
        return Ok(());
    }
    let script = format!(
        "Expand-Archive -LiteralPath '{}' -DestinationPath '{}' -Force",
        zip.display().to_string().replace('\'', "''"),
        into.display().to_string().replace('\'', "''")
    );
    let ps = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .status();
    match ps {
        Ok(s) if s.success() => Ok(()),
        _ => Err(format!(
            "could not unpack {}: neither tar nor PowerShell's Expand-Archive succeeded. \
             Nothing was changed.",
            display_path(zip)
        )),
    }
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
        .map_err(|e| format!("could not make {} executable: {e}", display_path(path)))
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

/// Run the new binary's `--version` before anything is replaced. It proves
/// the download is a binary this machine can start — the right system, the
/// DLLs of a Windows bundle beside it — and the version it was meant to be.
fn check_starts(new_exe: &Path, version: &str) -> Result<(), String> {
    let output = std::process::Command::new(new_exe)
        .arg("--version")
        .output()
        .map_err(|e| {
            format!("the new binary does not start on this machine ({e}). Nothing was changed.")
        })?;
    let said = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() || !said.contains(version) {
        return Err(format!(
            "the new binary answered `--version` with {:?} (exit {}), not {version}. \
             Nothing was changed.",
            said.trim(),
            output.status
        ));
    }
    Ok(())
}

/// One file of the new version and where it goes.
#[derive(Debug, Clone, PartialEq)]
struct Move {
    from: PathBuf,
    to: PathBuf,
}

/// Every file at the top of `unpacked` goes into `dir` under its own name,
/// except the new binary, which takes the running binary's name: someone
/// who renamed it, or runs the bare `eullm-windows-x64.exe`, keeps the name
/// they use.
fn planned_moves(
    unpacked: &Path,
    new_exe_name: &str,
    exe_name: &str,
    dir: &Path,
) -> Result<Vec<Move>, String> {
    let entries = std::fs::read_dir(unpacked)
        .map_err(|e| format!("could not read {}: {e}", display_path(unpacked)))?;
    let mut moves = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("could not read the unpacked files: {e}"))?;
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let name = entry.file_name();
        let target = if name == new_exe_name {
            exe_name.to_string()
        } else {
            name.to_string_lossy().into_owned()
        };
        moves.push(Move {
            from: entry.path(),
            to: dir.join(target),
        });
    }
    if !moves
        .iter()
        .any(|m| m.to.file_name().is_some_and(|n| n == exe_name))
    {
        return Err(format!(
            "the download holds no {new_exe_name}. Nothing was changed."
        ));
    }
    moves.sort_by(|a, b| a.to.cmp(&b.to));
    Ok(moves)
}

fn old_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(OLD_SUFFIX);
    path.with_file_name(name)
}

/// The `*.old` files an earlier update could not delete because they were
/// still in use. Only those of files this update replaces: nothing else in
/// the directory is touched.
fn remove_old_leftovers(moves: &[Move]) {
    for m in moves {
        let old = old_path(&m.to);
        if old.exists() {
            let _ = std::fs::remove_file(&old);
        }
    }
}

/// Put every file in place, renaming what it replaces to `*.old` first.
/// If any step fails, everything done so far is undone, so the directory is
/// left as it was. Returns how many old files could not be deleted afterwards
/// (on Windows: the running exe and the DLLs it loaded).
fn swap(moves: &[Move]) -> Result<usize, String> {
    let mut set_aside: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut placed: Vec<PathBuf> = Vec::new();

    let mut step = || -> Result<(), String> {
        for m in moves {
            if m.to.exists() {
                let old = old_path(&m.to);
                if old.exists() {
                    std::fs::remove_file(&old)
                        .map_err(|e| format!("{} is still in use ({e})", display_path(&old)))?;
                }
                std::fs::rename(&m.to, &old)
                    .map_err(|e| format!("could not set {} aside: {e}", display_path(&m.to)))?;
                set_aside.push((m.to.clone(), old));
            }
            std::fs::rename(&m.from, &m.to)
                .map_err(|e| format!("could not put {} in place: {e}", display_path(&m.to)))?;
            placed.push(m.to.clone());
        }
        Ok(())
    };

    if let Err(e) = step() {
        for path in placed.iter().rev() {
            let _ = std::fs::remove_file(path);
        }
        for (original, old) in set_aside.iter().rev() {
            let _ = std::fs::rename(old, original);
        }
        return Err(format!("{e}. The previous version was put back."));
    }

    Ok(set_aside
        .iter()
        .filter(|(_, old)| std::fs::remove_file(old).is_err())
        .count())
}

/// A Microsoft Store (MSIX) install lives under `WindowsApps`, which the
/// Store owns and updates.
fn is_store_install(exe: &Path) -> bool {
    exe.components()
        .any(|c| c.as_os_str().eq_ignore_ascii_case("WindowsApps"))
}

/// A path without the `\\?\` prefix `canonicalize` adds on Windows.
fn display_path(path: &Path) -> String {
    let s = path.display().to_string();
    s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The Windows ROCm ZIP is the one asset in ROCm's folder layout; every
    // other release asset is flat and stays updatable.
    #[test]
    fn only_the_windows_rocm_zip_keeps_rocms_layout() {
        assert!(keeps_rocm_layout("eullm-windows-x64-rocm.zip"));
        for asset in [
            "eullm-windows-x64.zip",
            "eullm-windows-x64-vulkan.zip",
            "eullm-windows-x64-cuda-13.1.zip",
            "eullm-linux-x64-rocm-consumer",
        ] {
            assert!(!keeps_rocm_layout(asset), "{asset}");
        }
    }

    #[test]
    fn versions_parse_as_the_tags_write_them() {
        assert_eq!(parse_version("0.7.40"), Some((0, 7, 40)));
        assert_eq!(parse_version("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_version("0.7.40-rc1"), None);
        assert_eq!(parse_version("0.7"), None);
        assert_eq!(parse_version("0.7.40.1"), None);
        assert_eq!(parse_version("latest"), None);
        // Numbers, not text: 0.7.100 is newer than 0.7.90.
        assert!(parse_version("0.7.100") > parse_version("0.7.90"));
        assert!(parse_version("0.8.0") > parse_version("0.7.90"));
    }

    #[test]
    fn the_running_version_parses() {
        assert!(running_version().is_some());
    }

    #[test]
    fn the_tag_is_read_from_the_redirect() {
        assert_eq!(
            tag_from_location("https://github.com/eullm/eullm/releases/tag/EuLLM-v0.7.40"),
            Some("EuLLM-v0.7.40")
        );
        assert_eq!(
            tag_from_location("/eullm/eullm/releases/tag/EuLLM-v0.7.40/"),
            Some("EuLLM-v0.7.40")
        );
        assert_eq!(
            tag_from_location("https://github.com/eullm/eullm/releases"),
            None
        );
        assert_eq!(
            tag_from_location("https://github.com/eullm/eullm/releases/tag/"),
            None
        );
    }

    #[test]
    fn checksums_are_matched_on_the_asset_name() {
        let h1 = "a".repeat(64);
        let h2 = "b".repeat(64);
        let text = format!(
            "{h1}  eullm-linux-x64\n{h2}  eullm-windows-x64-vulkan/eullm-windows-x64-vulkan.zip\n"
        );
        assert_eq!(checksum_for(&text, "eullm-linux-x64"), Some(h1.clone()));
        assert_eq!(
            checksum_for(&text, "eullm-windows-x64-vulkan.zip"),
            Some(h2)
        );
        // A prefix of another asset's name is not that asset.
        assert_eq!(checksum_for(&text, "eullm-linux"), None);
        assert_eq!(
            checksum_for("short  eullm-linux-x64\n", "eullm-linux-x64"),
            None
        );
    }

    #[test]
    fn a_store_install_is_recognised_by_its_folder() {
        assert!(
            is_store_install(Path::new(
                r"C:\Program Files\WindowsApps\EuLLM_0.7.40.0_x64__abc\eullm.exe"
            )) || cfg!(not(windows))
        );
        assert!(is_store_install(Path::new("/x/WindowsApps/eullm")));
        assert!(!is_store_install(Path::new("/home/me/.local/bin/eullm")));
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("eullm-update-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_new_binary_takes_the_running_binarys_name() {
        let root = scratch("names");
        let unpacked = root.join("files");
        std::fs::create_dir_all(&unpacked).unwrap();
        std::fs::write(unpacked.join("eullm.exe"), b"new").unwrap();
        std::fs::write(unpacked.join("msvcp140.dll"), b"dll").unwrap();
        let moves = planned_moves(&unpacked, "eullm.exe", "eullm-windows-x64.exe", &root).unwrap();
        let targets: Vec<_> = moves
            .iter()
            .map(|m| m.to.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(targets, ["eullm-windows-x64.exe", "msvcp140.dll"]);
        // A download without the binary is refused before anything moves.
        std::fs::remove_file(unpacked.join("eullm.exe")).unwrap();
        assert!(planned_moves(&unpacked, "eullm.exe", "eullm.exe", &root).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_swap_replaces_every_file_and_leaves_nothing_old_where_it_can() {
        let root = scratch("swap");
        let unpacked = root.join("files");
        std::fs::create_dir_all(&unpacked).unwrap();
        std::fs::write(root.join("eullm"), b"old binary").unwrap();
        std::fs::write(root.join("models.txt"), b"not ours").unwrap();
        std::fs::write(unpacked.join("eullm"), b"new binary").unwrap();
        std::fs::write(unpacked.join("libextra.so"), b"new lib").unwrap();
        let moves = planned_moves(&unpacked, "eullm", "eullm", &root).unwrap();
        let leftovers = swap(&moves).unwrap();
        assert_eq!(std::fs::read(root.join("eullm")).unwrap(), b"new binary");
        assert_eq!(std::fs::read(root.join("libextra.so")).unwrap(), b"new lib");
        assert_eq!(std::fs::read(root.join("models.txt")).unwrap(), b"not ours");
        if cfg!(unix) {
            assert_eq!(leftovers, 0);
            assert!(!root.join("eullm.old").exists());
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_failed_swap_puts_the_previous_version_back() {
        let root = scratch("rollback");
        let unpacked = root.join("files");
        std::fs::create_dir_all(&unpacked).unwrap();
        std::fs::write(root.join("a"), b"old a").unwrap();
        std::fs::write(root.join("b"), b"old b").unwrap();
        std::fs::write(unpacked.join("a"), b"new a").unwrap();
        // `b`'s new file is missing, so its move fails after `a` was placed.
        let moves = vec![
            Move {
                from: unpacked.join("a"),
                to: root.join("a"),
            },
            Move {
                from: unpacked.join("b"),
                to: root.join("b"),
            },
        ];
        let err = swap(&moves).unwrap_err();
        assert!(err.contains("previous version was put back"), "{err}");
        assert_eq!(std::fs::read(root.join("a")).unwrap(), b"old a");
        assert_eq!(std::fs::read(root.join("b")).unwrap(), b"old b");
        assert!(!root.join("a.old").exists() && !root.join("b.old").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn leftovers_of_an_earlier_update_are_cleared_only_for_replaced_files() {
        let root = scratch("leftovers");
        std::fs::write(root.join("eullm.old"), b"older").unwrap();
        std::fs::write(root.join("notes.old"), b"someone else's").unwrap();
        let moves = vec![Move {
            from: root.join("new"),
            to: root.join("eullm"),
        }];
        remove_old_leftovers(&moves);
        assert!(!root.join("eullm.old").exists());
        assert!(root.join("notes.old").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
