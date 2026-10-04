//! EuLLM: the llama.cpp changes this crate carries as patch files.
//!
//! `patches/*.patch` hold changes EuLLM needs in llama.cpp before upstream has
//! them: git diffs against the commit the submodule is pinned to, applied in
//! file-name order. They are applied to a copy of the submodule in `OUT_DIR`,
//! and everything the build script builds reads that copy: CMake, the wrapper
//! sources, mtmd and bindgen. A patch may therefore change any file, headers
//! included, and the submodule itself is never written to: `git status` stays
//! clean and `git submodule update` never conflicts.
//!
//! The copy keeps each file's modification time, so CMake recompiles what
//! changed and nothing else, and a patched file is rewritten only when its
//! patched content changes. A stamp of the patches and of the submodule's file
//! sizes and times skips the whole step when neither moved.
//!
//! Applying is strict and needs no `git`: every line a hunk expects must
//! match, at the line the hunk names or at the nearest offset past the
//! previous hunk. Lines are compared without their `\r`, so a Windows checkout
//! that turned both the sources and the patches into CRLF applies the same.
//! When a llama.cpp bump makes a patch stop applying, the build fails naming
//! the patch, the file and the hunk: regenerate the patch against the new pin,
//! or drop it once upstream has the change.
//!
//! Without patches the submodule and the wrapper sources are built where they
//! are, as upstream llama-cpp-rs does.
//!
//! The tests are in `engine/tests/llama_patches.rs`: `cargo test` never runs
//! a build script's own.

use std::collections::hash_map::DefaultHasher;
use std::collections::{btree_map, BTreeMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

/// llama.cpp directories nothing here builds from, left out of the copy: the
/// test vocabularies, the documentation, and what the build script turns off
/// (examples, server, web UI). The web UI's paths are also the ones that, under
/// a Windows OUT_DIR, would pass the 260-character limit.
const SKIPPED_DIRS: &[&str] = &[
    "models",
    "docs",
    "media",
    "examples",
    "tools/server",
    "tools/ui",
];

/// The wrapper sources. With patches they are copied beside the patched tree,
/// so that their `#include "llama.cpp/..."` lines read it.
const WRAPPER_FILES: &[&str] = &[
    "wrapper.h",
    "wrapper_common.h",
    "wrapper_common.cpp",
    "wrapper_utils.h",
    "wrapper_mtmd.h",
];

/// What the build script builds from.
pub struct Sources {
    /// The llama.cpp tree: the submodule, or its patched copy.
    pub llama_src: PathBuf,
    /// Where the wrapper sources are read from.
    pub wrapper_dir: PathBuf,
    /// The patches applied, by file name; empty when the submodule is built as is.
    pub patches: Vec<String>,
}

/// A patch file, by name and content.
pub struct PatchFile {
    pub name: String,
    pub text: String,
}

/// One file's changes in a patch.
#[derive(Debug)]
pub struct FileChange {
    pub path: String,
    pub new_file: bool,
    pub hunks: Vec<Hunk>,
}

/// One hunk: the lines it expects and the lines it leaves, context included.
#[derive(Debug)]
pub struct Hunk {
    pub header: String,
    pub old_start: usize,
    pub old: Vec<String>,
    pub new: Vec<String>,
}

/// A text file as lines without their line ends, and how to write it back.
#[derive(Debug)]
pub struct TextFile {
    pub lines: Vec<String>,
    crlf: bool,
    final_newline: bool,
}

/// A file of a tree, by its path from the root with `/` separators.
struct TreeFile {
    rel: String,
    len: u64,
    modified: SystemTime,
}

/// Applies `manifest_dir/patches/*.patch` to a copy of `submodule` in
/// `out_dir`, and says what to build from. Panics, naming the patch, the file
/// and the hunk, when a patch does not apply.
pub fn prepare(manifest_dir: &Path, submodule: &Path, out_dir: &Path) -> Sources {
    let patches_dir = manifest_dir.join("patches");
    // a path that does not exist would rerun the build script on every build;
    // the directory is kept, with its README, so that a first patch is seen
    if patches_dir.is_dir() {
        println!("cargo:rerun-if-changed={}", patches_dir.display());
    }
    let patches = read_patches(&patches_dir);
    if patches.is_empty() {
        return Sources {
            llama_src: submodule.to_path_buf(),
            wrapper_dir: manifest_dir.to_path_buf(),
            patches: Vec::new(),
        };
    }
    for patch in &patches {
        println!(
            "cargo:rerun-if-changed={}",
            patches_dir.join(&patch.name).display()
        );
    }

    for name in WRAPPER_FILES {
        copy_keeping_time(&manifest_dir.join(name), &out_dir.join(name));
    }
    let copy = out_dir.join("llama.cpp");
    patch_tree(submodule, &patches, &copy, &out_dir.join("llama.cpp.stamp"));
    Sources {
        llama_src: copy,
        wrapper_dir: out_dir.to_path_buf(),
        patches: patches.into_iter().map(|p| p.name).collect(),
    }
}

/// The `*.patch` files of `dir`, in name order; none if `dir` does not exist.
pub fn read_patches(dir: &Path) -> Vec<PatchFile> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "patch"))
                .collect()
        })
        .unwrap_or_default();
    paths.sort();
    paths
        .iter()
        .map(|p| PatchFile {
            name: p.file_name().unwrap().to_string_lossy().into_owned(),
            text: fs::read_to_string(p)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display())),
        })
        .collect()
}

/// Makes `copy` the `submodule` tree with `patches` applied, writing only the
/// files that differ from what `copy` holds; `stamp_path` records what it was
/// made from, so that the next call with the same inputs does nothing.
/// Panics, naming the patch, the file and the hunk, when a patch does not apply.
pub fn patch_tree(submodule: &Path, patches: &[PatchFile], copy: &Path, stamp_path: &Path) {
    let files = list_files(submodule);
    let stamp = stamp(patches, &files);
    if copy.is_dir() && fs::read_to_string(stamp_path).is_ok_and(|s| s == stamp) {
        return;
    }
    let _ = fs::remove_file(stamp_path);

    let patched = apply_patches(submodule, patches).unwrap_or_else(|e| panic!("{e}"));

    let mut kept = HashSet::new();
    for file in &files {
        kept.insert(file.rel.clone());
        let dst = copy.join(&file.rel);
        if !patched.contains_key(&file.rel) && !is_copy_of(&dst, file) {
            copy_keeping_time(&submodule.join(&file.rel), &dst);
        }
    }
    for (rel, text) in &patched {
        kept.insert(rel.clone());
        let dst = copy.join(rel);
        let content = text.render();
        if fs::read(&dst).ok().as_deref() != Some(content.as_bytes()) {
            create_parent(&dst);
            fs::write(&dst, content)
                .unwrap_or_else(|e| panic!("cannot write {}: {e}", dst.display()));
        }
    }
    // a file the submodule no longer has, or that is no longer copied, must
    // go: CMake globs some directories
    remove_others(copy, "", &kept);
    fs::write(stamp_path, stamp)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", stamp_path.display()));
}

/// The files `patches` change, as they read once all are applied, by path.
pub fn apply_patches(
    submodule: &Path,
    patches: &[PatchFile],
) -> Result<BTreeMap<String, TextFile>, String> {
    let mut patched: BTreeMap<String, TextFile> = BTreeMap::new();
    for patch in patches {
        let changes = parse_patch(&patch.text).map_err(|e| format!("{}: {e}", patch.name))?;
        if changes.is_empty() {
            return Err(format!("{}: no file changes in it", patch.name));
        }
        for change in changes {
            let file = match patched.entry(change.path.clone()) {
                btree_map::Entry::Occupied(e) => e.into_mut(),
                btree_map::Entry::Vacant(e) => {
                    let original = submodule.join(&change.path);
                    let file = if change.new_file {
                        if original.exists() {
                            return Err(format!(
                                "{} creates {}, which llama.cpp already has",
                                patch.name, change.path
                            ));
                        }
                        TextFile::new_file()
                    } else {
                        let text = fs::read_to_string(&original).map_err(|e| {
                            format!(
                                "{} changes {}, which cannot be read: {e}",
                                patch.name, change.path
                            )
                        })?;
                        TextFile::parse(&text)
                    };
                    e.insert(file)
                }
            };
            apply_hunks(&mut file.lines, &change.hunks).map_err(|e| {
                format!(
                    "{} does not apply to llama.cpp's {}: {e}. The patches are diffs against the \
                     commit the llama.cpp submodule is pinned to: when the pin moves, regenerate \
                     them against the new commit, or drop the ones upstream now has.",
                    patch.name, change.path
                )
            })?;
        }
    }
    Ok(patched)
}

/// Reads the file changes of a git diff. Text before the first `diff --git`
/// line, such as a description of the patch, is ignored.
pub fn parse_patch(text: &str) -> Result<Vec<FileChange>, String> {
    let mut lines: Vec<&str> = text
        .split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect();
    // what follows the last line end is not a line
    if text.ends_with('\n') {
        lines.pop();
    }
    let mut changes = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if !lines[i].starts_with("diff --git ") {
            i += 1;
            continue;
        }
        let section = lines[i];
        i += 1;

        let mut path = None;
        let mut new_file = false;
        while i < lines.len() && !lines[i].starts_with("@@") && !lines[i].starts_with("diff --git ")
        {
            let line = lines[i];
            if line.starts_with("new file mode") {
                new_file = true;
            } else if [
                "deleted file mode",
                "rename ",
                "copy ",
                "Binary files",
                "GIT binary patch",
            ]
            .iter()
            .any(|p| line.starts_with(p))
            {
                return Err(format!("{section}: unsupported change: {line}"));
            } else if let Some(p) = line.strip_prefix("+++ ") {
                path = Some(p.strip_prefix("b/").unwrap_or(p).to_string());
            }
            i += 1;
        }
        let path = path.ok_or_else(|| format!("{section}: no +++ line"))?;
        // the file is written under the copy, and nowhere else
        if path.is_empty()
            || path.starts_with('/')
            || path.contains('\\')
            || path.contains(':')
            || path
                .split('/')
                .any(|c| c.is_empty() || c == "." || c == "..")
        {
            return Err(format!(
                "{section}: {path:?} is not a path inside llama.cpp"
            ));
        }

        let mut hunks = Vec::new();
        while i < lines.len() && lines[i].starts_with("@@") {
            let header = lines[i];
            let (old_start, old_count, new_count) = parse_hunk_header(header)
                .ok_or_else(|| format!("{path}: unreadable hunk header {header}"))?;
            i += 1;
            let mut old = Vec::new();
            let mut new = Vec::new();
            while old.len() < old_count || new.len() < new_count {
                let line = *lines
                    .get(i)
                    .ok_or_else(|| format!("{path}: hunk {header} ends early"))?;
                i += 1;
                match line.chars().next() {
                    // an empty line is context whose leading space an editor dropped
                    Some(' ') | None => {
                        let context = line.get(1..).unwrap_or("");
                        old.push(context.to_string());
                        new.push(context.to_string());
                    }
                    Some('-') => old.push(line[1..].to_string()),
                    Some('+') => new.push(line[1..].to_string()),
                    Some('\\') => {}
                    _ => return Err(format!("{path}: unexpected line in hunk {header}: {line}")),
                }
            }
            // "\ No newline at end of file" after the hunk's last line
            while i < lines.len() && lines[i].starts_with('\\') {
                i += 1;
            }
            if old.len() != old_count || new.len() != new_count {
                return Err(format!("{path}: hunk {header} has more lines than it says"));
            }
            hunks.push(Hunk {
                header: header.to_string(),
                old_start,
                old,
                new,
            });
        }
        if hunks.is_empty() && !new_file {
            return Err(format!("{path}: a change with no hunks"));
        }
        changes.push(FileChange {
            path,
            new_file,
            hunks,
        });
    }
    Ok(changes)
}

/// `@@ -12,7 +12,9 @@ ...` as (12, 7, 9); a count left out is 1.
fn parse_hunk_header(header: &str) -> Option<(usize, usize, usize)> {
    let (ranges, _) = header.strip_prefix("@@ -")?.split_once(" @@")?;
    let (old, new) = ranges.split_once(" +")?;
    let range = |r: &str| -> Option<(usize, usize)> {
        match r.split_once(',') {
            Some((start, count)) => Some((start.parse().ok()?, count.parse().ok()?)),
            None => Some((r.parse().ok()?, 1)),
        }
    };
    let (old_start, old_count) = range(old)?;
    let (_, new_count) = range(new)?;
    Some((old_start, old_count, new_count))
}

/// Applies `hunks`, in order, to `lines`.
pub fn apply_hunks(lines: &mut Vec<String>, hunks: &[Hunk]) -> Result<(), String> {
    let mut shift: isize = 0;
    let mut floor = 0;
    for hunk in hunks {
        // `-5,0` inserts after line 5; `-5,3` replaces from line 5
        let named = if hunk.old.is_empty() {
            hunk.old_start
        } else {
            hunk.old_start.saturating_sub(1)
        };
        let expected = (named as isize + shift).max(0) as usize;
        let at = find_block(lines, &hunk.old, expected, floor)
            .ok_or_else(|| format!("hunk {} does not match", hunk.header))?;
        lines.splice(at..at + hunk.old.len(), hunk.new.iter().cloned());
        // the next hunk is expected as far from where it is named as this one was
        shift = at as isize - named as isize + hunk.new.len() as isize - hunk.old.len() as isize;
        floor = at + hunk.new.len();
    }
    Ok(())
}

/// Where `block` starts in `lines`, at `floor` or after, nearest `expected` first.
fn find_block(lines: &[String], block: &[String], expected: usize, floor: usize) -> Option<usize> {
    let last = lines.len().checked_sub(block.len())?;
    if floor > last {
        return None;
    }
    let expected = expected.clamp(floor, last);
    let fits = |at: usize| lines[at..at + block.len()] == *block;
    for distance in 0..=(last - floor) {
        if expected >= floor + distance && fits(expected - distance) {
            return Some(expected - distance);
        }
        if distance > 0 && expected + distance <= last && fits(expected + distance) {
            return Some(expected + distance);
        }
    }
    None
}

impl TextFile {
    fn new_file() -> Self {
        TextFile {
            lines: Vec::new(),
            crlf: false,
            final_newline: true,
        }
    }

    /// Splits `text` into lines, remembering its line ends.
    pub fn parse(text: &str) -> Self {
        let body = text.strip_suffix('\n').unwrap_or(text);
        TextFile {
            lines: if text.is_empty() {
                Vec::new()
            } else {
                body.split('\n')
                    .map(|l| l.strip_suffix('\r').unwrap_or(l).to_string())
                    .collect()
            },
            crlf: text.contains("\r\n"),
            final_newline: text.is_empty() || text.ends_with('\n'),
        }
    }

    /// The text again, with the line ends it was read with.
    pub fn render(&self) -> String {
        let eol = if self.crlf { "\r\n" } else { "\n" };
        let mut text = self.lines.join(eol);
        if self.final_newline && !self.lines.is_empty() {
            text.push_str(eol);
        }
        text
    }
}

/// The build information CMake would read with git in the submodule, which
/// it cannot in the copy: there git finds the repository the build directory
/// sits in. `None` without git or a git checkout.
pub fn build_info(submodule: &Path, n_patches: usize) -> Option<(String, String)> {
    let git = |args: &[&str]| -> Option<String> {
        let out = Command::new("git")
            .arg("-C")
            .arg(submodule)
            .args(args)
            .output()
            .ok()?;
        let text = String::from_utf8(out.stdout).ok()?.trim().to_string();
        (out.status.success() && !text.is_empty()).then_some(text)
    };
    let commit = git(&["rev-parse", "--short", "HEAD"])?;
    let number = git(&["rev-list", "--count", "HEAD"])?;
    Some((format!("{commit}+{n_patches}patches"), number))
}

/// CMake refuses a build directory configured from another source directory,
/// which is what moving between the submodule and its patched copy does, and
/// the compiler's dependency files there still name the old headers: start
/// that build directory afresh.
pub fn reset_build_dir_if_moved(build_dir: &Path, src: &Path) {
    let Ok(cache) = fs::read_to_string(build_dir.join("CMakeCache.txt")) else {
        return;
    };
    let Some(configured) = cache
        .lines()
        .find_map(|l| l.strip_prefix("CMAKE_HOME_DIRECTORY:INTERNAL="))
    else {
        return;
    };
    let same = match (fs::canonicalize(configured), fs::canonicalize(src)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same {
        println!(
            "cargo:warning=llama.cpp now builds from {}: rebuilding it from scratch",
            src.display()
        );
        let _ = fs::remove_dir_all(build_dir);
    }
}

fn stamp(patches: &[PatchFile], files: &[TreeFile]) -> String {
    let mut hasher = DefaultHasher::new();
    for patch in patches {
        patch.name.hash(&mut hasher);
        patch.text.hash(&mut hasher);
    }
    for file in files {
        file.rel.hash(&mut hasher);
        file.len.hash(&mut hasher);
        file.modified.hash(&mut hasher);
    }
    format!(
        "{:016x}: {} patches on {} files\n",
        hasher.finish(),
        patches.len(),
        files.len()
    )
}

/// The files under `root`, in name order, leaving out hidden entries and
/// `SKIPPED_DIRS`.
fn list_files(root: &Path) -> Vec<TreeFile> {
    let mut files = Vec::new();
    walk(root, "", &mut files);
    files
}

fn walk(dir: &Path, prefix: &str, files: &mut Vec<TreeFile>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(Result::ok).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let rel = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        if name.starts_with('.') || SKIPPED_DIRS.contains(&rel.as_str()) {
            continue;
        }
        let Ok(meta) = fs::metadata(entry.path()) else {
            continue;
        };
        if meta.is_dir() {
            walk(&entry.path(), &rel, files);
        } else if meta.is_file() {
            files.push(TreeFile {
                rel,
                len: meta.len(),
                modified: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            });
        }
    }
}

/// Removes from `dir` every file `kept` does not name, by its path from the
/// copy's root, and the directories that leaves empty; says whether `dir` is
/// empty now.
fn remove_others(dir: &Path, prefix: &str, kept: &HashSet<String>) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    let mut empty = true;
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name().to_string_lossy().into_owned();
        let rel = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        let path = entry.path();
        let Ok(meta) = fs::symlink_metadata(&path) else {
            empty = false;
            continue;
        };
        if meta.is_dir() {
            if remove_others(&path, &rel, kept) {
                let _ = fs::remove_dir(&path);
            } else {
                empty = false;
            }
        } else if kept.contains(&rel) || fs::remove_file(&path).is_err() {
            empty = false;
        }
    }
    empty
}

fn is_copy_of(dst: &Path, file: &TreeFile) -> bool {
    fs::metadata(dst).is_ok_and(|m| {
        m.is_file() && m.len() == file.len && m.modified().ok() == Some(file.modified)
    })
}

fn copy_keeping_time(src: &Path, dst: &Path) {
    create_parent(dst);
    fs::copy(src, dst)
        .unwrap_or_else(|e| panic!("cannot copy {} to {}: {e}", src.display(), dst.display()));
    // the source's time: CMake then sees the copy change only when the source does
    if let Ok(modified) = fs::metadata(src).and_then(|m| m.modified()) {
        let _ = fs::File::options()
            .write(true)
            .open(dst)
            .and_then(|f| f.set_modified(modified));
    }
}

fn create_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("cannot create {}: {e}", parent.display()));
    }
}
