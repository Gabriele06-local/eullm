//! The token methods `LlamaModel` had until llama-cpp-rs 0.1.158 moved them to
//! `LlamaVocab`, kept as they were so that the engine's ~35 call sites do not
//! change with the crate: an extension trait over `LlamaModel`, each method a
//! thin layer on `model.vocab()`.
//!
//! Two differences from the old methods, both in the crate's favour: text is
//! tokenized as bytes, so a NUL inside it is no longer an error, and a token
//! whose piece is empty decodes to an empty string instead of
//! `UnknownTokenType` (llama-server does the same; the old error ended a
//! generation on such a token with "decode failed"). The error types stay for
//! the signatures and can no longer be built.

use std::num::NonZeroU16;

use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token_type::LlamaTokenAttrs;

/// Whether to let the tokenizer add the model's BOS token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddBos {
    Always,
    Never,
}

/// Tokenizing bytes cannot fail; never built.
#[derive(Debug)]
pub enum StringToTokenError {}

impl std::fmt::Display for StringToTokenError {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {}
    }
}

impl std::error::Error for StringToTokenError {}

/// Decoding a token cannot fail; never built.
#[derive(Debug)]
pub enum TokenToStringError {}

impl std::fmt::Display for TokenToStringError {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {}
    }
}

impl std::error::Error for TokenToStringError {}

/// The old `LlamaModel` token methods, over the model's vocabulary.
pub trait ModelTokens {
    /// Tokenize `text`, special-token text such as `<|im_end|>` becoming the
    /// control token.
    fn str_to_token(&self, text: &str, add_bos: AddBos)
    -> Result<Vec<LlamaToken>, StringToTokenError>;

    /// Tokenize `text` with special-token text left as text: what a user
    /// wrote must be tokenized as the ordinary text it is.
    fn str_to_token_plain(
        &self,
        text: &str,
        add_bos: AddBos,
    ) -> Result<Vec<LlamaToken>, StringToTokenError>;

    /// One token as text, through a stateful UTF-8 decoder: a character split
    /// across tokens is held in `decoder` until its last byte arrives.
    fn token_to_piece(
        &self,
        token: LlamaToken,
        decoder: &mut encoding_rs::Decoder,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<String, TokenToStringError>;

    /// One token as the bytes the model wrote. `buffer_size` is the room to
    /// start with; the buffer grows when the piece does not fit.
    fn token_to_piece_bytes(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<Vec<u8>, TokenToStringError>;

    /// Whether `token` ends a generation (EOS, EOT and the like).
    fn is_eog_token(&self, token: LlamaToken) -> bool;

    /// The end-of-sequence token.
    fn token_eos(&self) -> LlamaToken;

    /// The attributes of `token` (control, unknown, byte...).
    fn token_attr(&self, token: LlamaToken) -> LlamaTokenAttrs;
}

impl ModelTokens for LlamaModel {
    fn str_to_token(
        &self,
        text: &str,
        add_bos: AddBos,
    ) -> Result<Vec<LlamaToken>, StringToTokenError> {
        Ok(self
            .vocab()
            .tokenize(text.as_bytes(), add_bos == AddBos::Always, true))
    }

    fn str_to_token_plain(
        &self,
        text: &str,
        add_bos: AddBos,
    ) -> Result<Vec<LlamaToken>, StringToTokenError> {
        Ok(self
            .vocab()
            .tokenize(text.as_bytes(), add_bos == AddBos::Always, false))
    }

    fn token_to_piece(
        &self,
        token: LlamaToken,
        decoder: &mut encoding_rs::Decoder,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<String, TokenToStringError> {
        let bytes = self.vocab().token_to_piece(token, special, lstrip);
        let mut text = String::with_capacity(
            decoder
                .max_utf8_buffer_length(bytes.len())
                .unwrap_or(bytes.len() * 3 + 4),
        );
        // Invalid bytes become U+FFFD, and the capacity above holds the
        // whole input, so one call consumes it.
        let _ = decoder.decode_to_string(&bytes, &mut text, false);
        Ok(text)
    }

    fn token_to_piece_bytes(
        &self,
        token: LlamaToken,
        buffer_size: usize,
        special: bool,
        lstrip: Option<NonZeroU16>,
    ) -> Result<Vec<u8>, TokenToStringError> {
        let mut buffer = Vec::with_capacity(buffer_size.max(8));
        self.vocab()
            .token_to_piece_into(token, &mut buffer, special, lstrip);
        Ok(buffer)
    }

    fn is_eog_token(&self, token: LlamaToken) -> bool {
        self.vocab().is_eog(token)
    }

    fn token_eos(&self) -> LlamaToken {
        self.vocab().eos()
    }

    fn token_attr(&self, token: LlamaToken) -> LlamaTokenAttrs {
        self.vocab().attr(token)
    }
}
