use crate::lex::lex;
use crate::parse::parse;
use crate::eval::{eval, Environment, EvalResult};

/// Lexes, parses, and evaluates the given program.
pub fn run_interpreter(program: &str) -> EvalResult {
    // TODO
    match lex(&program){
        Err(err) => return EvalResult::Err(format!("Lex error: {:?}", err)),
        Ok(tokens) => match parse(&tokens) {
            Err(err) => return EvalResult::Err(format!("Parse error: {:?}", err)),
            Ok(expr) => {
                let mut env = Environment::default();
                return eval(expr.clone(), &mut env)

            }
        }
    }
}
