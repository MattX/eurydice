pub mod ast;
pub mod dice;
pub mod eval;
pub mod output;

use lalrpop_util::lalrpop_mod;
lalrpop_mod!(pub grammar);
