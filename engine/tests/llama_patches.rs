//! Tests for the module the llama.cpp build script applies EuLLM's llama.cpp
//! patches with (`vendor/llama-cpp-rs/llama-cpp-sys-2/llama_patches.rs`).
//! `cargo test` never runs a build script's own tests, so they are here.

#[allow(dead_code)]
#[path = "../vendor/llama-cpp-rs/llama-cpp-sys-2/llama_patches.rs"]
mod llama_patches;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use llama_patches::{apply_patches, parse_patch, patch_tree, read_patches, PatchFile, TextFile};

/// A directory of its own under the system temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "eullm-llama-patches-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write(path: &Path, text: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

fn patch(name: &str, text: &str) -> PatchFile {
    PatchFile {
        name: name.to_string(),
        text: text.to_string(),
    }
}

fn modified(path: &Path) -> SystemTime {
    fs::metadata(path).unwrap().modified().unwrap()
}

fn set_modified(path: &Path, time: SystemTime) {
    fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(time)
        .unwrap();
}

const SOURCE: &str = "one\ntwo\nthree\nfour\nfive\nsix\nseven\n";

/// `two`..`six` with `four` replaced, as git writes it, after a description.
const CHANGE_FOUR: &str = "\
A description of the change, which is not part of the diff.

diff --git a/src/a.cpp b/src/a.cpp
index 1111111..2222222 100644
--- a/src/a.cpp
+++ b/src/a.cpp
@@ -2,5 +2,6 @@
 two
 three
-four
+FOUR
+four and a half
 five
 six
";

#[test]
fn a_hunk_replaces_its_lines_where_it_says() {
    let changes = parse_patch(CHANGE_FOUR).unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].path, "src/a.cpp");
    assert!(!changes[0].new_file);

    let mut file = TextFile::parse(SOURCE);
    llama_patches::apply_hunks(&mut file.lines, &changes[0].hunks).unwrap();
    assert_eq!(
        file.render(),
        "one\ntwo\nthree\nFOUR\nfour and a half\nfive\nsix\nseven\n"
    );
}

#[test]
fn a_hunk_still_applies_where_upstream_moved_its_lines() {
    let changes = parse_patch(CHANGE_FOUR).unwrap();
    let mut file = TextFile::parse(&format!("added upstream\nand again\n{SOURCE}"));
    llama_patches::apply_hunks(&mut file.lines, &changes[0].hunks).unwrap();
    assert_eq!(
        file.render(),
        "added upstream\nand again\none\ntwo\nthree\nFOUR\nfour and a half\nfive\nsix\nseven\n"
    );
}

#[test]
fn a_hunk_whose_lines_upstream_changed_names_itself() {
    let changes = parse_patch(CHANGE_FOUR).unwrap();
    let mut file = TextFile::parse(&SOURCE.replace("three", "3"));
    let err = llama_patches::apply_hunks(&mut file.lines, &changes[0].hunks).unwrap_err();
    assert!(err.contains("@@ -2,5 +2,6 @@"), "{err}");
}

#[test]
fn crlf_sources_and_patches_apply_and_stay_crlf() {
    let changes = parse_patch(&CHANGE_FOUR.replace('\n', "\r\n")).unwrap();
    let mut file = TextFile::parse(&SOURCE.replace('\n', "\r\n"));
    llama_patches::apply_hunks(&mut file.lines, &changes[0].hunks).unwrap();
    assert_eq!(
        file.render(),
        "one\r\ntwo\r\nthree\r\nFOUR\r\nfour and a half\r\nfive\r\nsix\r\nseven\r\n"
    );
}

#[test]
fn a_file_without_a_final_newline_keeps_it_missing() {
    let text = "one\ntwo\nthree";
    let file = TextFile::parse(text);
    assert_eq!(file.lines, ["one", "two", "three"]);
    assert_eq!(file.render(), text);
    assert_eq!(TextFile::parse("").render(), "");
}

#[test]
fn hunk_counts_left_out_mean_one_line() {
    let patch = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -3 +3 @@\n-three\n+THREE\n";
    let changes = parse_patch(patch).unwrap();
    let mut file = TextFile::parse(SOURCE);
    llama_patches::apply_hunks(&mut file.lines, &changes[0].hunks).unwrap();
    assert_eq!(file.lines[2], "THREE");
}

#[test]
fn malformed_and_unsupported_patches_are_refused() {
    // the hunk says 5 old lines and has 4
    let short = CHANGE_FOUR.replace(" six\n", "");
    assert!(parse_patch(&short).unwrap_err().contains("ends early"));
    let renamed = "diff --git a/x b/y\nsimilarity index 100%\nrename from x\nrename to y\n";
    assert!(parse_patch(renamed).unwrap_err().contains("unsupported"));
    let deleted = "diff --git a/x b/x\ndeleted file mode 100644\n--- a/x\n+++ /dev/null\n";
    assert!(parse_patch(deleted).unwrap_err().contains("unsupported"));
    // a patch writes inside the copy of llama.cpp, nowhere else
    for path in ["../x", "src/../../x", "/etc/x", "C:/x", "src//x"] {
        let outside = format!(
            "diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-a\n+b\n"
        );
        let err = parse_patch(&outside).unwrap_err();
        assert!(err.contains("not a path inside llama.cpp"), "{path}: {err}");
    }
}

#[test]
fn patches_build_on_each_other_and_may_create_files() {
    let tmp = TempDir::new();
    let submodule = tmp.0.join("llama.cpp");
    write(&submodule.join("src/a.cpp"), SOURCE);

    let create_b = "\
diff --git a/src/b.cpp b/src/b.cpp
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/src/b.cpp
@@ -0,0 +1,2 @@
+int b();
+int b() { return 0; }
";
    let second = format!(
        "\
diff --git a/src/a.cpp b/src/a.cpp
--- a/src/a.cpp
+++ b/src/a.cpp
@@ -4,3 +4,3 @@
 FOUR
-four and a half
+four and three quarters
 five
{create_b}"
    );
    let patched = apply_patches(
        &submodule,
        &[
            patch("0001-a.patch", CHANGE_FOUR),
            patch("0002-b.patch", &second),
        ],
    )
    .unwrap();
    assert_eq!(
        patched["src/a.cpp"].render(),
        "one\ntwo\nthree\nFOUR\nfour and three quarters\nfive\nsix\nseven\n"
    );
    assert_eq!(
        patched["src/b.cpp"].render(),
        "int b();\nint b() { return 0; }\n"
    );

    // creating a file llama.cpp already has is a patch written for another tree
    write(&submodule.join("src/b.cpp"), "upstream\n");
    let err = apply_patches(&submodule, &[patch("0002-b.patch", create_b)]).unwrap_err();
    assert!(err.contains("already has"), "{err}");
}

#[test]
fn a_patch_that_does_not_apply_names_the_patch_and_the_file() {
    let tmp = TempDir::new();
    let submodule = tmp.0.join("llama.cpp");
    write(&submodule.join("src/a.cpp"), &SOURCE.replace("five", "5"));
    let err = apply_patches(&submodule, &[patch("0001-a.patch", CHANGE_FOUR)]).unwrap_err();
    assert!(err.contains("0001-a.patch"), "{err}");
    assert!(err.contains("src/a.cpp"), "{err}");
    assert!(err.contains("regenerate"), "{err}");
}

#[test]
fn the_copy_rewrites_only_what_changed() {
    let tmp = TempDir::new();
    let submodule = tmp.0.join("llama.cpp");
    write(&submodule.join("src/a.cpp"), SOURCE);
    write(&submodule.join("src/kept.cpp"), "kept\n");
    write(&submodule.join("CMakeLists.txt"), "project(x)\n");
    write(&submodule.join("models/vocab.gguf"), "big\n");
    write(&submodule.join(".github/workflow.yml"), "ci\n");
    write(&submodule.join("tools/ui/src/App.svelte"), "<p>ui</p>\n");
    write(&submodule.join("tools/mtmd/mtmd.cpp"), "mtmd\n");
    let an_hour_ago = SystemTime::now() - Duration::from_secs(3600);
    set_modified(&submodule.join("src/kept.cpp"), an_hour_ago);

    let copy = tmp.0.join("out/llama.cpp");
    let stamp = tmp.0.join("out/llama.cpp.stamp");
    let patches = [patch("0001-a.patch", CHANGE_FOUR)];
    patch_tree(&submodule, &patches, &copy, &stamp);

    assert_eq!(
        fs::read_to_string(copy.join("src/a.cpp")).unwrap(),
        "one\ntwo\nthree\nFOUR\nfour and a half\nfive\nsix\nseven\n"
    );
    assert_eq!(
        fs::read_to_string(copy.join("src/kept.cpp")).unwrap(),
        "kept\n"
    );
    // a copy keeps its source's time, so CMake does not rebuild it
    assert_eq!(modified(&copy.join("src/kept.cpp")), an_hour_ago);
    assert!(copy.join("CMakeLists.txt").is_file());
    assert!(!copy.join("models").exists());
    assert!(!copy.join(".github").exists());
    assert!(!copy.join("tools/ui").exists());
    assert!(copy.join("tools/mtmd/mtmd.cpp").is_file());

    // the patched file is rewritten only when its patched content changes
    let patched_at = an_hour_ago - Duration::from_secs(60);
    set_modified(&copy.join("src/a.cpp"), patched_at);
    set_modified(&submodule.join("CMakeLists.txt"), an_hour_ago);
    patch_tree(&submodule, &patches, &copy, &stamp);
    assert_eq!(modified(&copy.join("src/a.cpp")), patched_at);
    assert_eq!(modified(&copy.join("CMakeLists.txt")), an_hour_ago);

    // a file the submodule dropped leaves the copy, with a directory it leaves
    // empty, and so does one a copy made before a directory was skipped holds;
    // a new file joins it
    fs::remove_file(submodule.join("src/kept.cpp")).unwrap();
    fs::remove_dir_all(submodule.join("tools/mtmd")).unwrap();
    write(&submodule.join("src/new.cpp"), "new\n");
    write(&copy.join("examples/old/old.cpp"), "old\n");
    patch_tree(&submodule, &patches, &copy, &stamp);
    assert!(!copy.join("src/kept.cpp").exists());
    assert!(!copy.join("tools").exists());
    assert!(!copy.join("examples").exists());
    assert_eq!(
        fs::read_to_string(copy.join("src/new.cpp")).unwrap(),
        "new\n"
    );

    // without the patch, the file is the submodule's again
    patch_tree(&submodule, &[patch("0001-other.patch", "diff --git a/src/new.cpp b/src/new.cpp\n--- a/src/new.cpp\n+++ b/src/new.cpp\n@@ -1 +1 @@\n-new\n+newer\n")], &copy, &stamp);
    assert_eq!(fs::read_to_string(copy.join("src/a.cpp")).unwrap(), SOURCE);
    assert_eq!(
        fs::read_to_string(copy.join("src/new.cpp")).unwrap(),
        "newer\n"
    );
}

#[test]
fn the_patches_carried_apply_to_the_pinned_llama_cpp() {
    let crate_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/llama-cpp-rs/llama-cpp-sys-2");
    let submodule = crate_dir.join("llama.cpp");
    if !submodule.join("include/llama.h").is_file() {
        // the build needs the submodule anyway; nothing to check without it
        return;
    }
    let patches = read_patches(&crate_dir.join("patches"));
    if let Err(e) = apply_patches(&submodule, &patches) {
        panic!("{e}");
    }
}
