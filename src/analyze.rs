//! A simple static analysis pass.
//! 
//! Checks that:
//!  * Functions are in scope and called with the right arity
//!  * Refernces are valid
//!  * `output` and `set` statements do not appear inside functions
