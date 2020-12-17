use std::env;
use std::fs;
use mlisp::interpreter::run_interpreter;
use mlisp::lex::lex;
use mlisp::parse::parse;
use mlisp::eval::{eval, Environment, EvalResult};


fn main() {
    let args: Vec<String> = env::args().collect();
    println!("args: {:?}", args);
    assert!(args.len() > 1, "Must supply a file path.");
    let content = fs::read_to_string(&args[1])
        .expect("There was an error reading the file");

    println!("Read content: {}", content);
    run_interpreter(&content);

    match lex(&content){
        Err(err) => println!("Lex error: {:?}", err),
        Ok(tokens) => match parse(&tokens) {
            Err(err) => println!("Parse error: {:?}", err),
            Ok(expr) => {
                let mut env = Environment::default();
                match eval(expr.clone(), &mut env) {
                    EvalResult::Err(err) => println!("{}", err),
                    _ => {}, 
                }
            }
        }
    }
}
