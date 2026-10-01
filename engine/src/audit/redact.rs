//! Personal data taken out of text before it is written to a decision trace
//! (`api::decision_traces`): what a decision was about is kept, who it was
//! about is not. Six kinds of data are recognised, each replaced by a typed
//! placeholder so the text still reads as what it was:
//!
//! | Placeholder | What | Recognised by |
//! |---|---|---|
//! | `[EMAIL]` | e-mail addresses | `name@domain.tld`, international letters included |
//! | `[IBAN]` | IBANs of any country | country, check digits and account, whole, in groups of four or by its parts; the mod-97 check must pass |
//! | `[CF]` | Italian codici fiscali | the 16-character structure, omocodia included, with a valid month and day; not the check letter, so a mistyped one is caught too |
//! | `[CARD]` | payment card numbers | 13 to 19 digits starting with 2 to 6, whole or in the groups cards are printed in; the Luhn check must pass |
//! | `[IP]` | IPv4 addresses | four numbers from 0 to 255 joined by dots |
//! | `[PHONE]` | phone numbers | international (`+` or `00`, any country) and Italian: a mobile, ten digits from a 3; a landline, 8 to 11 digits from a 0 |
//!
//! It is pattern matching, not understanding. Names, street addresses,
//! dates of birth, IPv6 addresses and every identifier not in the table
//! stay as written, as does anything in the table written in a way it does
//! not expect: a spaced-out codice fiscale, `mario at example dot com`, a
//! foreign number without its `+`. And it errs in the other direction too: a
//! version string `1.2.3.4` reads as an address, and a 13–19 digit code
//! that happens to pass the Luhn check — one in ten do — as a card.
//!
//! Dates, amounts, article numbers (`art. 2043 c.c.`), years and ordinary
//! numbers are left alone, and the tests below hold examples of each.

/// `text` with every e-mail address, IBAN, codice fiscale, card number,
/// IPv4 address and phone number replaced by its placeholder.
///
/// The order matters. E-mail addresses go first, since a local part may
/// hold any of the others' digits (`mario.3331234567@…`); IBANs before
/// cards and phones, whose digit runs they contain; and a placeholder holds
/// no digit, `@` or run of 16 letters, so no later pass finds anything in
/// one.
pub fn redact(text: &str) -> String {
    let text = replace(text, "[EMAIL]", email_at);
    let text = replace(&text, "[IBAN]", iban_at);
    let text = replace(&text, "[CF]", codice_fiscale_at);
    let text = replace(&text, "[CARD]", card_at);
    let text = replace(&text, "[IP]", ipv4_at);
    replace(&text, "[PHONE]", phone_at)
}

/// `text` with every match of `matcher` replaced by `placeholder`.
/// `matcher(text, i)` is asked at each character boundary `i` outside an
/// earlier match, and returns where a match starting there ends.
fn replace(text: &str, placeholder: &str, matcher: fn(&str, usize) -> Option<usize>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    let mut i = 0;
    while i < text.len() {
        match matcher(text, i) {
            Some(end) => {
                out.push_str(&text[copied..i]);
                out.push_str(placeholder);
                copied = end;
                i = end;
            }
            None => i += text[i..].chars().next().map_or(1, char::len_utf8),
        }
    }
    out.push_str(&text[copied..]);
    out
}

/// Whether the two characters before byte `i` are a JSON escape: `\n`,
/// `\r` or `\t`. Structured instructions and descriptions reach a trace as
/// JSON text, where a line break before a phone number is those two
/// characters, and the letter must not hide the number.
fn after_escape(text: &str, i: usize) -> bool {
    let mut before = text[..i].chars().rev();
    matches!(before.next(), Some('n' | 'r' | 't')) && before.next() == Some('\\')
}

/// Whether a word may start at byte `i`: at the start of the text, after a
/// character that is not a letter or a digit, or after a JSON escape.
fn starts_word(text: &str, i: usize) -> bool {
    after_escape(text, i)
        || !text[..i]
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric)
}

/// Whether a word may end at byte `i`: nothing follows, or something that
/// is not a letter or a digit.
fn ends_word(text: &str, i: usize) -> bool {
    !text[i..].chars().next().is_some_and(char::is_alphanumeric)
}

/// [`starts_word`], and not in the middle of a larger number: not right
/// after `12.`, `12,`, `12/` or `12-`.
fn starts_number(text: &str, i: usize) -> bool {
    if !starts_word(text, i) {
        return false;
    }
    let mut before = text[..i].chars().rev();
    !matches!(
        (before.next(), before.next()),
        (Some('.' | ',' | '/' | '-'), Some(d)) if d.is_ascii_digit()
    )
}

/// [`ends_word`], and not followed by more of the same number: not by
/// `.5` or `,50`.
fn ends_number(text: &str, i: usize) -> bool {
    let b = text.as_bytes();
    ends_word(text, i)
        && !(matches!(b.get(i), Some(b'.' | b',')) && b.get(i + 1).is_some_and(u8::is_ascii_digit))
}

// ── E-mail addresses ────────────────────────────────────────────────────

fn is_local(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-')
}

fn is_domain(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '.' | '-')
}

/// `name@domain.tld`: a local part, and a domain of at least two labels
/// whose last is two or more letters. A trailing dot or hyphen is the
/// sentence's, not the address's.
fn email_at(text: &str, i: usize) -> Option<usize> {
    let rest = &text[i..];
    let first = rest.chars().next()?;
    let after_local = text[..i].chars().next_back().is_some_and(is_local);
    if !is_local(first)
        || first == '.'
        || (after_local && !after_escape(text, i))
        // The letter of an escape is not the address's: it starts after it.
        || (matches!(first, 'n' | 'r' | 't') && text[..i].ends_with('\\'))
    {
        return None;
    }
    let at = rest.find(|c: char| !is_local(c))?;
    if !rest[at..].starts_with('@') {
        return None;
    }
    let host = &rest[at + 1..];
    let host_len = host.find(|c: char| !is_domain(c)).unwrap_or(host.len());
    let domain = host[..host_len].trim_end_matches(['.', '-']);
    let labels: Vec<&str> = domain.split('.').collect();
    let tld = labels.last()?;
    let valid = labels.len() >= 2
        && labels
            .iter()
            .all(|l| !l.is_empty() && !l.starts_with('-') && !l.ends_with('-'))
        && tld.chars().count() >= 2
        && tld.chars().all(char::is_alphabetic);
    valid.then(|| i + at + 1 + domain.len())
}

// ── IBANs ───────────────────────────────────────────────────────────────

/// An IBAN: two letters, two check digits, then the account, 15 to 34
/// characters in all, whose mod-97 check passes. Written whole
/// (`IT60X0542811101000000123456`), in groups of four
/// (`IT60 X054 2811 1010 0000 0123 456`), or by its parts, as Italian
/// documents print it (`IT 60 X 05428 11101 000000123456`).
fn iban_at(text: &str, i: usize) -> Option<usize> {
    if !starts_word(text, i) {
        return None;
    }
    iban_in_fours(text, i).or_else(|| iban_by_parts(text, i))
}

/// An IBAN whole or in groups of four.
fn iban_in_fours(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    let head = b.get(i..i + 4)?;
    let country_and_check = head[0].is_ascii_alphabetic()
        && head[1].is_ascii_alphabetic()
        && head[2].is_ascii_digit()
        && head[3].is_ascii_digit();
    if !country_and_check {
        return None;
    }
    // Runs of letters and digits; a run of four may be followed by a space
    // and the next, as an IBAN is printed. `ends` holds, after each run, its
    // end in `text` and the IBAN's length so far.
    let mut iban = String::new();
    let mut ends = Vec::new();
    let mut j = i;
    loop {
        let start = j;
        while j < b.len() && b[j].is_ascii_alphanumeric() {
            j += 1;
        }
        iban.push_str(&text[start..j]);
        ends.push((j, iban.len()));
        let grouped = j - start == 4
            && b.get(j) == Some(&b' ')
            && b.get(j + 1).is_some_and(u8::is_ascii_alphanumeric);
        if !grouped || iban.len() > 34 {
            break;
        }
        j += 1;
    }
    longest_iban(text, &iban, &ends)
}

/// An IBAN by its parts: the country, the check digits — each maybe
/// followed by a space — then parts of any length, each with a digit in it
/// or a single letter (Italy's CIN), one of them five characters or more,
/// at least ten digits in all. Those conditions, and the check, keep a
/// sentence that starts with a short word and a number, or a list of
/// numbers, from reading as one.
fn iban_by_parts(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    if !(b.get(i)?.is_ascii_alphabetic() && b.get(i + 1)?.is_ascii_alphabetic()) {
        return None;
    }
    let mut j = i + 2;
    if b.get(j) == Some(&b' ') {
        j += 1;
    }
    if !(b.get(j)?.is_ascii_digit() && b.get(j + 1)?.is_ascii_digit()) {
        return None;
    }
    let mut iban = format!("{}{}", &text[i..i + 2], &text[j..j + 2]);
    j += 2;
    let mut ends = Vec::new();
    let mut longest_part = 0;
    while b.get(j) == Some(&b' ') && iban.len() <= 34 {
        let start = j + 1;
        let mut k = start;
        while k < b.len() && b[k].is_ascii_alphanumeric() {
            k += 1;
        }
        let part = &text[start..k];
        let fits = part.bytes().any(|c| c.is_ascii_digit())
            || (part.len() == 1 && part.as_bytes()[0].is_ascii_alphabetic());
        if !fits || !ends_word(text, k) {
            break;
        }
        iban.push_str(part);
        ends.push((k, iban.len()));
        longest_part = longest_part.max(part.len());
        j = k;
    }
    let digits = iban.bytes().filter(u8::is_ascii_digit).count();
    (digits >= 10 && longest_part >= 5).then(|| longest_iban(text, &iban, &ends))?
}

/// The end of the longest IBAN among `iban`'s prefixes that end at `ends`,
/// longest first: the word after an IBAN whose last group is a full four,
/// or a code after it, was read as one more group.
fn longest_iban(text: &str, iban: &str, ends: &[(usize, usize)]) -> Option<usize> {
    ends.iter()
        .rev()
        .find(|&&(end, len)| {
            (15..=34).contains(&len) && ends_word(text, end) && iban_checks(&iban[..len])
        })
        .map(|&(end, _)| end)
}

/// The IBAN check: the first four characters moved to the end, letters
/// read as 10 to 35, the number modulo 97 is 1. Check digits 00, 01 and 99
/// are never issued.
fn iban_checks(iban: &str) -> bool {
    let b = iban.as_bytes();
    let check = &iban[2..4];
    if matches!(check, "00" | "01" | "99") {
        return false;
    }
    let mut rest = 0u32;
    for &c in b[4..].iter().chain(&b[..4]) {
        rest = match c {
            b'0'..=b'9' => (rest * 10 + u32::from(c - b'0')) % 97,
            b'a'..=b'z' | b'A'..=b'Z' => {
                (rest * 100 + u32::from(c.to_ascii_uppercase() - b'A') + 10) % 97
            }
            _ => return false,
        };
    }
    rest == 1
}

// ── Codici fiscali ──────────────────────────────────────────────────────

/// Letters standing for the digits 0–9 in a codice fiscale changed for
/// omocodia, when two people would otherwise share one.
const OMOCODIA: &[u8; 10] = b"LMNPQRSTUV";

/// Month letters, January to December.
const CF_MONTHS: &[u8; 12] = b"ABCDEHLMPRST";

/// A codice fiscale, `RSSMRA80A01H501U`: six letters for the surname and
/// name, the year, a month letter, the day (plus 40 for women), a letter and
/// three digits for the place of birth, and a check letter.
fn codice_fiscale_at(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    let cf = b.get(i..i + 16)?;
    if !cf[0].is_ascii_alphabetic()
        || !cf.iter().all(u8::is_ascii_alphanumeric)
        || !starts_word(text, i)
        || !ends_word(text, i + 16)
    {
        return None;
    }
    is_codice_fiscale(cf).then_some(i + 16)
}

fn is_codice_fiscale(cf: &[u8]) -> bool {
    let cf: Vec<u8> = cf.iter().map(u8::to_ascii_uppercase).collect();
    let letter = |c: u8| c.is_ascii_uppercase();
    // A digit, or the letter that stands for it.
    let digit = |c: u8| {
        if c.is_ascii_digit() {
            Some(c - b'0')
        } else {
            OMOCODIA.iter().position(|&o| o == c).map(|d| d as u8)
        }
    };
    let day = digit(cf[9])
        .zip(digit(cf[10]))
        .map(|(tens, units)| tens * 10 + units);
    cf[..6].iter().all(|&c| letter(c))
        && digit(cf[6]).is_some()
        && digit(cf[7]).is_some()
        && CF_MONTHS.contains(&cf[8])
        && day.is_some_and(|d| (1..=31).contains(&d) || (41..=71).contains(&d))
        && letter(cf[11])
        && cf[12..15].iter().all(|&c| digit(c).is_some())
        && letter(cf[15])
}

// ── Payment cards ───────────────────────────────────────────────────────

/// A card number: 13 to 19 digits, the first 2 to 6 — every major network
/// — contiguous or in the groups cards are printed in, with a valid Luhn
/// check digit.
fn card_at(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    if !(b'2'..=b'6').contains(b.get(i)?) || !starts_number(text, i) {
        return None;
    }
    let (digits, groups) = digit_groups(text, i, b" -", 19);
    // Longest first: an expiry date or a number after the card was read as
    // one more group.
    (1..=groups.len()).rev().find_map(|k| {
        let len: usize = groups[..k].iter().map(|g| g.0).sum();
        let end = groups[..k].last()?.1;
        let printed = card_grouping(&groups[..k]);
        ((13..=19).contains(&len) && printed && ends_number(text, end) && luhn(&digits[..len]))
            .then_some(end)
    })
}

/// Contiguous, in fours (`4111 1111 1111 1111`, the last group shorter on
/// longer cards), or as American Express and Diners print them, 4-6-5 and
/// 4-6-4.
fn card_grouping(groups: &[(usize, usize)]) -> bool {
    let sizes: Vec<usize> = groups.iter().map(|g| g.0).collect();
    match sizes.split_last() {
        Some((_, [])) => true,
        Some((&last, rest)) => {
            (rest.iter().all(|&s| s == 4) && (1..=4).contains(&last))
                || sizes == [4, 6, 5]
                || sizes == [4, 6, 4]
        }
        None => false,
    }
}

/// The Luhn check: every second digit from the right doubled, the digits
/// summed, a multiple of ten.
fn luhn(digits: &str) -> bool {
    let sum: u32 = digits
        .bytes()
        .rev()
        .enumerate()
        .map(|(k, d)| {
            let d = u32::from(d - b'0');
            match (k % 2 == 1, d * 2) {
                (false, _) => d,
                (true, doubled) if doubled > 9 => doubled - 9,
                (true, doubled) => doubled,
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

/// The runs of digits from byte `i`, joined by one separator from
/// `separators` — the same one throughout — up to `max` digits: the digits
/// together, and per run its length and where it ends in `text`.
fn digit_groups(
    text: &str,
    i: usize,
    separators: &[u8],
    max: usize,
) -> (String, Vec<(usize, usize)>) {
    let b = text.as_bytes();
    let mut digits = String::new();
    let mut groups = Vec::new();
    let mut separator = None;
    let mut j = i;
    loop {
        let start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j == start || digits.len() + (j - start) > max {
            break;
        }
        digits.push_str(&text[start..j]);
        groups.push((j - start, j));
        match b.get(j) {
            Some(&s)
                if separators.contains(&s)
                    && separator.is_none_or(|sep| sep == s)
                    && b.get(j + 1).is_some_and(u8::is_ascii_digit) =>
            {
                separator = Some(s);
                j += 1;
            }
            _ => break,
        }
    }
    (digits, groups)
}

// ── IPv4 addresses ──────────────────────────────────────────────────────

/// Four numbers from 0 to 255 joined by dots, not part of a longer dotted
/// run (`1.2.3.4.5`).
fn ipv4_at(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    if !b.get(i)?.is_ascii_digit() || !starts_number(text, i) {
        return None;
    }
    let mut j = i;
    for part in 0..4 {
        if part > 0 {
            if b.get(j) != Some(&b'.') {
                return None;
            }
            j += 1;
        }
        let start = j;
        while j < b.len() && b[j].is_ascii_digit() && j - start < 3 {
            j += 1;
        }
        if j == start || b.get(j).is_some_and(u8::is_ascii_digit) {
            return None;
        }
        // No leading zero: `3.000.000.000` is an amount, and an address is
        // not written `10.000.001.002`.
        if (j - start > 1 && b[start] == b'0') || text[start..j].parse::<u32>().ok()? > 255 {
            return None;
        }
    }
    ends_number(text, j).then_some(j)
}

// ── Phone numbers ───────────────────────────────────────────────────────

/// Most digits a phone number candidate collects: E.164's 15, a `00`
/// prefix and a trunk `(0)`.
const PHONE_MAX_DIGITS: usize = 18;

/// A run of digit groups that may be a phone number, as written.
struct PhoneRun {
    /// It started with `+`.
    plus: bool,
    digits: String,
    /// Per group, its digits and where it ends in `text`.
    groups: Vec<(usize, usize)>,
}

/// A phone number: international — `+39 333 1234567`, `0039 06 1234 5678`,
/// `+44 20 7946 0958`, `+1 (202) 555-0123` — or Italian without its
/// prefix: a mobile (`333 1234567`, `333-123-4567`) or a landline
/// (`06 1234 5678`, `(02) 1234567`, `0461 123456`).
fn phone_at(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    let first = *b.get(i)?;
    if !(first.is_ascii_digit() || first == b'+' || first == b'(') || !starts_number(text, i) {
        return None;
    }
    // A number that starts at its `+` was tried there.
    if first.is_ascii_digit() && text[..i].ends_with('+') {
        return None;
    }
    let run = phone_run(text, i)?;
    // Longest first: a number after the phone number, or the year after it,
    // was read as more of it.
    (1..=run.groups.len()).rev().find_map(|k| {
        let groups = &run.groups[..k];
        let len: usize = groups.iter().map(|g| g.0).sum();
        let end = groups[k - 1].1;
        (ends_number(text, end) && is_phone(run.plus, groups, &run.digits[..len])).then_some(end)
    })
}

/// The digit groups from byte `i`, each maybe in parentheses (`(02)`,
/// `(0)`, `(+39)`), the first maybe after a `+`. One separator joins two
/// groups: a space, a no-break space, `-`, `.` or `/`, or nothing after a
/// parenthesis. The separator after the first group and after one in
/// parentheses may be any of them, the others must all be the same — so
/// `+39 333-123-4567` is one number, while a date and the phone number after
/// it (`01/10/2026 06 12345678`) are two runs.
fn phone_run(text: &str, i: usize) -> Option<PhoneRun> {
    let b = text.as_bytes();
    let mut run = PhoneRun {
        plus: false,
        digits: String::new(),
        groups: Vec::new(),
    };
    let mut rule: Option<u8> = None;
    let mut j = i;
    loop {
        let mut k = j;
        let open = b.get(k) == Some(&b'(');
        if open {
            k += 1;
        }
        if run.groups.is_empty() && b.get(k) == Some(&b'+') {
            run.plus = true;
            k += 1;
        }
        let start = k;
        while k < b.len() && b[k].is_ascii_digit() {
            k += 1;
        }
        let count = k - start;
        if count == 0
            || run.digits.len() + count > PHONE_MAX_DIGITS
            || (open && b.get(k) != Some(&b')'))
        {
            break;
        }
        run.digits.push_str(&text[start..k]);
        if open {
            k += 1;
        }
        run.groups.push((count, k));
        j = k;

        // The separator to the next group, if one follows.
        let (separator, len) = match b.get(j) {
            Some(&c @ (b' ' | b'-' | b'.' | b'/')) => (Some(c), 1),
            // A no-break space is a space.
            Some(0xC2) if b.get(j + 1) == Some(&0xA0) => (Some(b' '), 2),
            Some(c) if open && (c.is_ascii_digit() || *c == b'(') => (None, 0),
            _ => break,
        };
        if !b
            .get(j + len)
            .is_some_and(|c| c.is_ascii_digit() || *c == b'(')
        {
            break;
        }
        if let Some(s) = separator
            && run.groups.len() > 1
            && !open
        {
            match rule {
                None => rule = Some(s),
                Some(r) if r != s => break,
                Some(_) => {}
            }
        }
        j += len;
    }
    (!run.groups.is_empty()).then_some(run)
}

/// Whether these groups, and their digits, are a phone number.
fn is_phone(plus: bool, groups: &[(usize, usize)], digits: &str) -> bool {
    let international = if plus {
        Some(digits)
    } else {
        digits.strip_prefix("00")
    };
    if let Some(number) = international {
        if number.starts_with('0') {
            return false;
        }
        // Italy's numbers by the Italian rules, which also keeps a second
        // number written after the first out of it.
        if let Some(national) = number.strip_prefix("39") {
            return italian_mobile(national, true) || italian_landline(national, true);
        }
        // Elsewhere, E.164's length. Written with `00` and no spaces, a
        // long code with leading zeros is likelier than a foreign number.
        return (8..=15).contains(&number.len()) && (plus || groups.len() > 1);
    }
    if looks_like_date(groups, digits) {
        return false;
    }
    match digits.as_bytes()[0] {
        // A mobile: the 3xx prefix, then the number, or all ten together.
        b'3' => italian_mobile(digits, false) && (groups.len() == 1 || groups[0].0 == 3),
        // A landline: the area code first. Eleven digits only when spaced
        // out: written whole, eleven digits from a 0 is a partita IVA.
        b'0' if groups.len() == 1 => italian_landline(digits, false) && digits.len() <= 10,
        b'0' => italian_landline(digits, false) && (2..=4).contains(&groups[0].0),
        _ => false,
    }
}

/// An Italian mobile number without its prefix: ten digits from a 3. Nine
/// on a few old lines, taken only after `+39`: without it, nine digits from
/// a 3 are as often an amount (`300.000.000`).
fn italian_mobile(national: &str, prefixed: bool) -> bool {
    let len = national.len();
    national.starts_with('3') && (len == 10 || (prefixed && len == 9))
}

/// An Italian landline without its prefix: from a 0 and an area code that
/// does not start with another 0, 6 to 11 digits after `+39`. Without it, 8
/// to 11: fewer are office hours (`08.30-12.30`) or a code as often as a
/// number in use.
fn italian_landline(national: &str, prefixed: bool) -> bool {
    let d = national.as_bytes();
    let shortest = if prefixed { 6 } else { 8 };
    d.len() >= shortest && d.len() <= 11 && d[0] == b'0' && d[1] != b'0'
}

/// `dd/mm/yyyy`, `dd.mm.yy` and `mm/yyyy`, which start with a 0 or a 3 as
/// often as a phone number does.
fn looks_like_date(groups: &[(usize, usize)], digits: &str) -> bool {
    let sizes: Vec<usize> = groups.iter().map(|g| g.0).collect();
    let number = |from: usize, len: usize| digits[from..from + len].parse::<u32>().unwrap_or(0);
    match sizes[..] {
        [d @ (1 | 2), m @ (1 | 2), 2 | 4] => {
            (1..=31).contains(&number(0, d)) && (1..=12).contains(&number(d, m))
        }
        [m @ (1 | 2), 4] => (1..=12).contains(&number(0, m)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each example is replaced, whole, by its placeholder, in a sentence.
    fn assert_redacted(placeholder: &str, examples: &[&str]) {
        for example in examples {
            let text = format!("Scrivimi: {example}, grazie.");
            assert_eq!(
                redact(&text),
                format!("Scrivimi: {placeholder}, grazie."),
                "{example}"
            );
        }
    }

    #[test]
    fn email_addresses() {
        assert_redacted(
            "[EMAIL]",
            &[
                "mario.rossi@example.com",
                "mario.rossi+fatture@pec.azienda.it",
                "M_Rossi-1985@sub.example.co.uk",
                "info@università.it",
                "josé@correo.es",
            ],
        );
        assert_eq!(
            redact("Write to mario@example.com."),
            "Write to [EMAIL].",
            "the sentence's full stop is not the address's"
        );
        assert_eq!(redact("(mailto:a.b@c.de)"), "(mailto:[EMAIL])");
        assert_eq!(redact("a@b@c.com"), "a@[EMAIL]");
        // After a JSON escape the address starts after its letter.
        assert_eq!(
            redact(r#"{"to":"Mario\nmario@example.com"}"#),
            r#"{"to":"Mario\n[EMAIL]"}"#
        );
    }

    #[test]
    fn what_only_looks_like_an_email_address_stays() {
        for text in [
            "@mario",
            "mario@",
            "mario@localhost",
            "x@y.z",
            "a@b.c1",
            "a@-b.com",
            "a@b..com",
            "costa 10 € @ pezzo",
            "@@",
        ] {
            assert_eq!(redact(text), text);
        }
    }

    #[test]
    fn ibans() {
        assert_redacted(
            "[IBAN]",
            &[
                "IT60X0542811101000000123456",
                "IT60 X054 2811 1010 0000 0123 456",
                "it60x0542811101000000123456",
                "DE89 3704 0044 0532 0130 00",
                "GB82 WEST 1234 5698 7654 32",
                "FR1420041010050500013M02606",
                "BE68 5390 0754 7034",
                // By its parts, as Italian documents print it.
                "IT 60 X 05428 11101 000000123456",
                "IT60 X 05428 11101 000000123456",
            ],
        );
        assert_eq!(
            redact("IBAN: IT 60 X 05428 11101 000000123456 presso Banca X"),
            "IBAN: [IBAN] presso Banca X"
        );
        // A word after an IBAN whose last group is a full four is not part
        // of it (a Spanish IBAN has 24 characters).
        assert_eq!(
            redact("IBAN ES91 2100 0418 4502 0005 1332 ABCD fine"),
            "IBAN [IBAN] ABCD fine"
        );
        // Whole groups that fail the check are not cut short to a length
        // that passes it.
        assert!(!redact("DE89 3704 0044 0532 0130 0012 3456").contains("[IBAN]"));
    }

    #[test]
    fn what_only_looks_like_an_iban_stays() {
        for text in [
            // Too short, or glued to a word.
            "IT60X05428111",
            "XIT60X0542811101000000123456",
            "AB12",
            "ISO 9001",
        ] {
            assert_eq!(redact(text), text);
        }
        // One character off, the check fails and it is no IBAN (its digit
        // groups could still read as a phone number).
        for text in [
            "IT60X0542811101000000123457",
            "IT60 X054 2811 1010 0000 0123 457",
            "IT 60 X 05428 11101 000000123457",
        ] {
            assert!(!redact(text).contains("[IBAN]"), "{text}");
        }
        // A short word and a number start many sentences.
        for text in [
            "da 10 anni lavoro presso la ditta",
            "Ho 12 anni e 3 mesi",
            "IT 60 X 5 sedie",
            "ID 12 3 4 5 6 7 8 9 10 11 12 13",
        ] {
            assert_eq!(redact(text), text);
        }
    }

    #[test]
    fn codici_fiscali() {
        assert_redacted(
            "[CF]",
            &[
                "RSSMRA80A01H501U",
                "rssmra80a01h501u",
                // A woman's (day + 40), omocodia, and a mistyped check letter.
                "BNCGNN75S52F205T",
                "RSSMRA80A01H5LMU",
                "RSSMRA80A01H501Z",
            ],
        );
        assert_eq!(redact("CF:RSSMRA80A01H501U."), "CF:[CF].");
    }

    #[test]
    fn what_only_looks_like_a_codice_fiscale_stays() {
        for text in [
            // Month letter Z, day 00, day 35, a digit where a letter goes.
            "RSSMRA80Z01H501U",
            "RSSMRA80A00H501U",
            "RSSMRA80A35H501U",
            "RSSMR180A01H501U",
            // Seventeen characters, fifteen, inside a word.
            "RSSMRA80A01H501UX",
            "RSSMRA80A01H501",
            "deadbeefcafebabe",
            "ABCDEFGHIJKLMNOP",
        ] {
            assert_eq!(redact(text), text);
        }
    }

    #[test]
    fn card_numbers() {
        assert_redacted(
            "[CARD]",
            &[
                "4111 1111 1111 1111",
                "4111-1111-1111-1111",
                "4111111111111111",
                "5500 0000 0000 0004",
                "3782 822463 10005",
                "6011111111111117",
                "4012888888881881",
                "4222222222222",
            ],
        );
        assert_eq!(
            redact("carta 4111 1111 1111 1111 scad. 12/27"),
            "carta [CARD] scad. 12/27"
        );
    }

    #[test]
    fn what_only_looks_like_a_card_number_stays() {
        for text in [
            // The Luhn check fails.
            "4111 1111 1111 1112",
            "4111111111111112",
            // Starts with 1, 7, 8, 9: no card network.
            "1234 5678 9012 3452",
            "7992739871300000",
            // Twenty digits, not a card printed in its groups.
            "41111111111111111111",
            "4111 11 1111 1111 1111",
        ] {
            assert_eq!(redact(text), text, "{text}");
        }
    }

    #[test]
    fn ipv4_addresses() {
        assert_redacted("[IP]", &["192.168.1.10", "10.0.0.1", "255.255.255.255"]);
        assert_eq!(redact("host 10.0.0.1:8080"), "host [IP]:8080");
        assert_eq!(redact("rete 10.1.0.0/16."), "rete [IP]/16.");
    }

    #[test]
    fn what_only_looks_like_an_ipv4_address_stays() {
        for text in [
            "256.1.1.1",
            "1.2.3",
            "1.2.3.4.5",
            "v1.2.3.4",
            "1.2.3.4a",
            "01.10.2026",
            "€ 3.000.000.000",
            "10.000.001.002",
        ] {
            assert_eq!(redact(text), text);
        }
    }

    #[test]
    fn phone_numbers() {
        assert_redacted(
            "[PHONE]",
            &[
                // International.
                "+39 333 1234567",
                "+393331234567",
                "+39 06 1234 5678",
                "(+39) 333 1234567",
                "0039 333 1234567",
                "00393331234567",
                "+44 20 7946 0958",
                "+1 (202) 555-0123",
                "+49 30 1234567",
                "0044 20 7946 0958",
                // A nine-digit mobile of an old line, with its prefix.
                "+39 335 123456",
                // Italian mobile.
                "333 1234567",
                "333-1234567",
                "333 123 4567",
                "333-123-4567",
                "333.123.4567",
                "333/1234567",
                "3331234567",
                "333\u{a0}1234567",
                // Italian landline.
                "06 1234 5678",
                "06 12345678",
                "06-12345678",
                "02/12345678",
                "(02) 12345678",
                "0461 123456",
                "011.1234567",
                "0612345678",
            ],
        );
        assert_eq!(redact("tel:3331234567"), "tel:[PHONE]");
        assert_eq!(redact("cell.3331234567"), "cell.[PHONE]");
        // Two numbers one after the other are two numbers.
        assert_eq!(
            redact("333 1234567 333 7654321"),
            "[PHONE] [PHONE]",
            "the second is not swallowed by the first"
        );
        assert_eq!(redact("+39 333 1234567 333 7654321"), "[PHONE] [PHONE]");
        assert_eq!(redact("+39 333-1234567"), "[PHONE]");
        // A date, then a phone number.
        assert_eq!(redact("il 01/10/2026 06 12345678"), "il 01/10/2026 [PHONE]");
        // The \n of JSON text does not hide the number after it.
        assert_eq!(
            redact(r#"{"note":"chiamare\n3331234567"}"#),
            r#"{"note":"chiamare\n[PHONE]"}"#
        );
    }

    /// The numbers a legal, administrative or business text is full of,
    /// none of them anyone's personal data.
    #[test]
    fn dates_amounts_articles_and_ordinary_numbers_stay() {
        for text in [
            "art. 2043 c.c.",
            "artt. 1175 e 1375 c.c.",
            "art. 360, comma 1, n. 3, c.p.c.",
            "d.lgs. 196/2003",
            "D.P.R. 445/2000",
            "legge 30 dicembre 2023, n. 213",
            "legge 300/1970",
            "Cass. civ., sez. III, 12/03/2021, n. 7024",
            "sentenza n. 31234/2019",
            "01/10/2026",
            "1/10/26",
            "06.10.2026",
            "31/12/2026",
            "03/04/2025",
            "03/2026",
            "2026-10-01",
            "2026-10-01T12:34:56Z",
            "ore 12:30",
            "dalle 08.30-12.30",
            "orario 08.30-12.30 / 14.00-18.00",
            "ABI 03069 CAB 09606",
            "€ 1.234,56",
            "1.250.000 euro",
            "€ 3.000",
            "€ 30.000.000",
            "€ 300.000.000",
            "€ 3.000.000.000",
            "12,5%",
            "+15%",
            "+3.5",
            "+45.4642, 9.1900",
            "ordine n. 123456",
            "ordine n. 312345678",
            "ordine 0012345",
            "00184 Roma",
            "20121 Milano",
            "dal 2019 al 2023",
            "2019-2023",
            "v0.7.20",
            "0.123456",
            "P.IVA 01234567890",
            "3 5 7 9 11 13 15 17 19 21",
            "Q4_K_M",
            "pagina 3 di 10",
            "42",
            "3,14",
            "10000",
        ] {
            assert_eq!(redact(text), text, "{text}");
        }
    }

    #[test]
    fn a_text_with_several_kinds_of_personal_data() {
        let text = "Sono Mario Rossi (RSSMRA80A01H501U), nato il 01/01/1980. Scrivete a \
                    mario.rossi@example.com o chiamate il +39 333 1234567 o lo 06 1234 5678. \
                    Rimborso su IT60 X054 2811 1010 0000 0123 456, non sulla carta \
                    4111 1111 1111 1111. Accesso da 192.168.1.10, ai sensi dell'art. 2043 c.c.";
        assert_eq!(
            redact(text),
            "Sono Mario Rossi ([CF]), nato il 01/01/1980. Scrivete a [EMAIL] o chiamate il \
             [PHONE] o lo [PHONE]. Rimborso su [IBAN], non sulla carta [CARD]. Accesso da \
             [IP], ai sensi dell'art. 2043 c.c."
        );
    }

    #[test]
    fn text_with_nothing_to_redact_is_unchanged() {
        for text in [
            "",
            "Help! My payouts have been failing for 3 days.",
            "Il contratto è nullo — l'art. 1418 c.c. è chiaro. 😀",
            "{\"subject\": \"Refund\", \"amount\": 12.5}",
        ] {
            assert_eq!(redact(text), text);
        }
    }

    mod properties {
        use super::super::*;
        use proptest::prelude::*;

        proptest! {
            /// Untrusted text of any shape: never a panic, and never a
            /// change to text that has no digit and no `@`.
            #[test]
            fn redaction_never_panics(text in ".*") {
                let out = redact(&text);
                if !text.chars().any(|c| c.is_ascii_digit() || c == '@') {
                    prop_assert_eq!(out, text);
                }
            }

            /// The same for text made of what the patterns look for.
            #[test]
            fn redaction_never_panics_on_number_like_text(
                text in "[0-9 +()./@a-zA-Z\\-\u{a0}\\\\]{0,64}"
            ) {
                let _ = redact(&text);
            }
        }
    }
}
