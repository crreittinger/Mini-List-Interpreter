use crate::types::Expr;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, PartialEq)]
pub enum EvalResult {
    Err(String),
    Expr(Rc<Expr>),
    Unit,
}

#[derive(Debug)]
pub struct Environment {
    pub contexts: Vec<HashMap<String, (Vec<String>, Rc<Expr>)>>,
}

impl Environment {
    pub fn empty() -> Environment {
        Environment {
            contexts: Vec::new(),
        }
    }

    /// Helper function for tests
    pub fn from_vars(vars: &[(&str, Rc<Expr>)]) -> Environment {
        let mut env = Environment::empty();
        env.push_context();
        vars.iter().for_each(|(name, expr)| {
            let _ = env.add_var(name, expr.clone());
        });
        env
    }
    pub fn default() -> Environment {
        let defaults: HashMap<String, (Vec<String>, Rc<Expr>)> = [
            ("False".into(), (Vec::new(), Expr::list(&[]))),
            ("True".into(), (Vec::new(), Expr::list(&[Expr::fnum(1.0)])))
        ].iter().cloned().collect();
        Environment{
            contexts: vec![defaults],
        }
    }

    /// Looks up the given symbol in the Environment.
    pub fn lookup(&self, symbol: &str) -> Option<(Vec<String>, Rc<Expr>)> {
        self.contexts
            .iter()
            .rev()
            .find(|ctx| ctx.contains_key(symbol))
            .map(|ctx|  ctx.get(symbol))
            .flatten()
            .cloned()
    }

    /// Checks whether the given symbol exists in the Environment.
    pub fn contains_key(&self, symbol: &str) -> bool {
       self.contexts
        .iter()
        .rev()
        .find(|ctx| ctx.contains_key(symbol))
        .is_some() 
    }

    /// Pushes a new context on the `contexts` stack.
    pub fn push_context(&mut self) {
        self.contexts.push(HashMap::new());
    }

    /// Pops the last context from the `contexts` stack.
    pub fn pop_context(&mut self) {
        self.contexts.pop();
    }

    /// Adds a variable definition to the Environment
    pub fn add_var(&mut self, var: &str, val: Rc<Expr>) -> Result<(), String> {
        self.contexts
            .last_mut()
            .map_or_else(
                || Err("Environment does not have a context to add to.".into()),
                |ctx| {ctx.insert(var.to_string(), (Vec::new(), val.clone())); Ok(()) }, 
            )
    }

    /// Adds a function definition to the Environment
    pub fn add_fn(&mut self, name: &str, params: &[String], body: Rc<Expr>) -> Result<(), String> {
        self.contexts.last_mut().map_or(
            Err("Environment does not have a context to add to.".into()),
            |ctx| {
                let param_names: Vec<String> = params.iter().map(|s| s.to_string()).collect();
                ctx.insert(name.to_string(), (param_names, body.clone()));
                Ok(())
            },
        )
    }

    pub fn num_contexts(&self) -> usize {
        self.contexts.len()
    }
}

/// Generates the output printed to standard out when the user calls print.
pub fn gen_print_output(expr: Rc<Expr>, env: &mut Environment) -> String {
    match &*expr {
        Expr::Symbol(s) => {
            match env.lookup(&s){
                //expr is a symbol
                None => s.to_string(),
                //expr is a variable 
                Some((params, e)) if params.len() == 0 => gen_print_output(e, env),
                //expr is a function
                _ => format!("<func-object: {}>", s.to_string()),
            }
        },
        Expr::FNum(n) => format!("{}",n),
        Expr::List(vals) => {
            let vals_out: Vec<String> = vals.iter()
                .cloned()
                .map(|x| gen_print_output(x, env))
                .collect();
            format!("({})", vals_out.join(" "))
        }
    }
}

fn evaluate_symbol(expr: Rc<Expr>, sym: &str, args: &[Rc<Expr>], env: &mut Environment) -> EvalResult{
    env.lookup(sym)
        .map_or_else(
            || EvalResult::Expr(expr),
            |(param_names, expression)| {
                if param_names.is_empty(){
                    eval(expression.clone(), env)
                }
                else{
                    
                    if args.len() != param_names.len(){
                        return EvalResult::Err(format!("provided {} arguments but expected {}", args.len(), param_names.len()));
                    }
                    let mapped_args: Result<Vec<(String, Rc<Expr>)>, String> = args.iter()
                        .zip(param_names)
                        .map(|(expr, name)|
                            match eval(expr.clone(), env){
                                EvalResult::Expr(e) => Ok((name.to_string(), e.clone())),
                                EvalResult::Err(err) => Err(err),
                                _ => Err("Cannot pass Unit as an argument to a funciton.".into()),
                            })
                        .collect();
                    env.push_context();
                    let result = mapped_args.map_or_else(
                        |e| EvalResult::Err(e),
                        |arg_tuples| {
                            arg_tuples.iter().for_each(|(name, expr)|{
                                 let _ =  env.add_var(name, expr.clone());
                            });
                            eval(expression.clone(), env)
  
                    });
                    env.pop_context();
                    result
                }
            }

        )
}

fn add_fn_to_env(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.len() != 3 {
        EvalResult::Err("Function definitions must follow the pattern (fn fn-name (arg1 arg2 arg3 .. argn) <Expr>".into());
    }
    let fn_name = &*vals[0];
    let p_names = &*vals[1];
    let body = &vals[2];
    match(&*fn_name, p_names, body) {
        (Expr::Symbol(fn_name), Expr::List(params), body) => {
            let ps: Result<Vec<String>, String> = params.iter().cloned().map(|e| {
                if let Expr::Symbol(n) = &*e {
                    Ok(n.to_string())
                }
                else{
                    Err("Function parameters must be symbols.".into())
                }
            })
            .collect();
            ps.map_or_else(
                |err| EvalResult::Err(err),
                |xs| env.add_fn(fn_name, xs.as_slice(), body.clone()).map_or_else(
                    |err| EvalResult::Err(err),
                    |_| EvalResult::Unit
                )
            )
        },
        _ => EvalResult::Err("Function definitions must follow the pattern (fn fn-name (arg1 arg2 arg3 .. argn) <Expr>".into()),
    }
}

/// Adds a variable to the environment from a let-statement
fn add_var_to_env(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    //Incorrectly defined let statement
    if vals.len() != 2 {
        return EvalResult::Err(
            "Invalid variable definition. Should look like (let someVar someExpr".into(),
        );
    }

    //Otherwise, let statement has correct number of elements
    match (&*vals[0], &vals[1]) {
        (Expr::Symbol(s), e) => 
        match eval(e.clone(), env) {
                EvalResult::Expr(e) => env
                    .add_var(s, e)
                    .map_or_else(
                        |s| EvalResult::Err(s),
                        |_| EvalResult::Unit,
                    ),
                EvalResult::Unit => EvalResult::Err("Cannot assign Unit to a variable".into()),
                err => err,
        },  
        _ => EvalResult::Err("Second element of variable def must be a symbol and third must be an expression".into(),
        ),
    }
}

fn not_op(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.len() != 1 {
        return EvalResult::Err("Must perform not operation on one expression.".into());
    }

    let first = &vals[0];
    if *first == Expr::symbol("True"){
        EvalResult::Expr(Expr::symbol("False"))
    }
    else if *first == Expr::symbol("False"){
        EvalResult::Expr(Expr::symbol("True"))
    }
    else{
        match eval(first.clone(), env) {
            EvalResult::Expr(expr) => {
                if expr == Expr::symbol("True") {
                    EvalResult::Expr(Expr::symbol("False"))
                }
                else if expr == Expr::symbol("False") {
                    EvalResult::Expr(Expr::symbol("True"))
                }
                else{
                    EvalResult::Err("Expression does not return True or False".into())
                }
            
             },
            err => err
        }
    }
    
}
fn and_check(vals: &[Rc<Expr>]) -> EvalResult {
    if vals.len() < 2 {
        return EvalResult::Err("Must perform and operation on at least 2 expressions.".into());
    }
    
    let first = &vals[0];
    if *first != Expr::symbol("True"){
        EvalResult::Expr(Expr::symbol("False"))
    }
    else{
        let x = vals.iter().all(|item| item == first);
        if x{
            EvalResult::Expr(Expr::symbol("True"))
        }
        else{
            EvalResult::Expr(Expr::symbol("False"))
        }
    }
    
}
fn or_check(vals: &[Rc<Expr>]) -> EvalResult {
    if vals.len() < 2 {
        return EvalResult::Err("Must perform or operation on at least 2 expressions.".into());
    }
    let first = &vals[0];
            if *first == Expr::symbol("True"){
                EvalResult::Expr(Expr::symbol("True"))
            }
            else{
                let x = vals.iter().all(|item| item == first);
                if x{
                    EvalResult::Expr(Expr::symbol("False"))
                }
                else{
                    EvalResult::Expr(Expr::symbol("True"))
                }
            }
}
fn equality_check(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.len() < 2 {
        return EvalResult::Err("Must perform equality check on at least 2 expressions.".into());
    }
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env){
            EvalResult::Expr(exp) => Ok(exp.clone()),
            _ => Err("Equality check can only be done on expressions".into())
        }).collect::<Result<Vec<Rc<Expr>>, String>>();
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| {
            let first = &xs[0];
            let check_equal = xs.iter().all(|x| x == first);
            if check_equal{
                EvalResult::Expr(Expr::symbol("True"))
            }
            else{
                EvalResult::Expr(Expr::symbol("False"))
            }
        }
    )
    
    


    
}
fn inequality_check(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.len() < 2 {
        return EvalResult::Err("Must perform inequality check on at least 2 expressions.".into());
    }
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env){
            EvalResult::Expr(exp) => Ok(exp.clone()),
            _ => Err("Inequality check can only be done on expressions".into())
        }).collect::<Result<Vec<Rc<Expr>>, String>>();

    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| {
            let first = &xs[0];
            let check_inequal = xs.iter().all(|x| x == first);
            if check_inequal{
                EvalResult::Expr(Expr::symbol("False"))
            }
            else{
                EvalResult::Expr(Expr::symbol("True"))
            }
        }
    )
}
fn mult_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Must perform multiplication on at least one number.".into());
    }

    let total = vals.iter()
        .map(|e| match eval(e.clone(), env){
            EvalResult::Expr(exp) => match &*exp {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only multiply numbers".into()),
            },
            _ => Err("Can only multiply numbers.".into())
        })
        .collect::<Result<Vec<f64>, String>>();

    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::fnum(xs.iter().product()))
    )
}

fn div_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Must perform division on at least one number.".into());
    }

    let total = vals.iter()
        .map(|e| match eval(e.clone(), env){
            EvalResult::Expr(exp) => match &*exp {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only divide numbers".into()),
            },
            _ => Err("Can only divide numbers.".into())
        })
        .collect::<Result<Vec<f64>, String>>();
    
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| {
            let first_val = xs[0];
            let xs_count = xs.len();
            if xs_count == 1{
                EvalResult::Expr(Expr::fnum(first_val))
            }
            else{
                let remaining_val = &xs[1..];
                let answer: f64 = remaining_val.iter().product();
                let r = first_val / answer;
                EvalResult::Expr(Expr::fnum(r))
            }
        }
    )
}

fn sub_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Must perform subtraction on at least one number.".into());
    }

    let total = vals.iter()
        .map(|e| match eval(e.clone(), env){
            EvalResult::Expr(exp) => match &*exp {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only subtract numbers".into()),
            },
            _ => Err("Can only subtract numbers.".into())
        })
        .collect::<Result<Vec<f64>, String>>();
    
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| {
            let first_val = xs[0];
            let xs_count = xs.len();
            if xs_count == 1{
                EvalResult::Expr(Expr::fnum(first_val))
            }
            else{
                let remaining_val = &xs[1..];
                let answer: f64 = remaining_val.iter().sum();
                let r = first_val - answer;
                EvalResult::Expr(Expr::fnum(r))
            }
        }
    )
}
//Addition operation: (+ 1 2 3 4 5)
fn add_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Must perform addition on at least one number.".into());
    }

    let total = vals.iter()
        .map(|e| match eval(e.clone(), env){
            EvalResult::Expr(exp) => match &*exp {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only sum numbers".into()),
            },
            _ => Err("Can only sum numbers.".into())
        })
        .collect::<Result<Vec<f64>, String>>();

    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::fnum(xs.iter().sum()))
    )
}

/// Implements if-then-else logic
/// Example: (if (<predicate-block>) (<then-block>) (<else-block>))
fn if_then_else(blocks: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if blocks.len() != 3 {
        return EvalResult::Err("If expressions must have the format (if (<predicate-block>) (<then-block>) (<else-block>))".into());
    }

    match eval(blocks[0].clone(), env) {
        EvalResult::Expr(expr) => {
           
            
            match &*expr{
                Expr::List(vs) if vs.len() == 0 => eval(blocks[2].clone(), env),
                _ => eval(blocks[1].clone(), env),
            }
        },
        EvalResult::Unit => EvalResult::Err("If expression predicates must return an expression.".into()),
        err => err
    }
}
/// Evaluates the given expression.
pub fn eval(e: Rc<Expr>, env: &mut Environment) -> EvalResult {
    match &*e {
        Expr::FNum(_) => EvalResult::Expr(e.clone()),
        Expr::Symbol(s) => evaluate_symbol(e.clone(), s, &[], env),
        Expr::List(vals) => {
            if vals.is_empty(){
                return EvalResult::Expr(Expr::list(&[]));
            }
            let op = &*vals[0]; 
            match op {

                //Case: (+ <Expr> <Expr> <Expr>)
                Expr::Symbol(s) if s == "+" => add_vals(&vals[1..], env), 

                Expr::Symbol(s) if s == "-" => sub_vals(&vals[1..], env),
                Expr::Symbol(s) if s == "*" => mult_vals(&vals[1..], env), 
                Expr::Symbol(s) if s == "/" => div_vals(&vals[1..], env),

                Expr::Symbol(s) if s == "=" => equality_check(&vals[1..], env),
                Expr::Symbol(s) if s == "!=" => inequality_check(&vals[1..], env),

                Expr::Symbol(s) if s == "not" => not_op(&vals[1..], env),


                Expr::Symbol(s) if s == "and" => and_check(&vals[1..]),
                Expr::Symbol(s) if s == "or" => or_check(&vals[1..]),
                //Case: (let x <Expr>)
                Expr::Symbol(s) if s == "let" => add_var_to_env(&vals[1..], env),

                //Case: (fn my-func (x1 x2 x3) <Expr>)
                Expr::Symbol(s) if s == "fn" => add_fn_to_env(&vals[1..], env),

                //Case: (print <expr> <expr> <expr> )
                Expr::Symbol(s) if s == "print" => {
                    let output: Vec<String> = vals[1..].iter()
                        .cloned()
                        .map(|expr| gen_print_output(expr, env))
                        .collect();
                    println!("{}",output.join(" "));
                    EvalResult::Unit
                },  
                Expr::Symbol(s) if s == "if" => if_then_else(&vals[1..], env),
                Expr::Symbol(s) if env.contains_key(&s)   => {
                    evaluate_symbol(e.clone(), s, &vals[1..], env)
                }
                _ => {
                    let res: Result<Vec<Rc<Expr>>, EvalResult> = vals.iter()
                        .cloned()
                        .map(|expr| eval(expr, env))
                        .filter(|x| *x != EvalResult::Unit)
                        .map(|x| if let EvalResult::Expr(expr) = x{
                            Ok(expr)
                        }
                        else{
                            Err(x)
                        })
                        .collect();
                    res.map_or_else(
                        |err| err,
                        |exprs| EvalResult::Expr(Expr::list(&exprs))
                    )
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn cannot_add_to_contextless_env(){
        let mut env = Environment::empty();

        let r = env.add_var("a", Expr::fnum(1.0));
        assert!(
            r.is_err(),
            format!("Expected add_var to fail, but it succeeded: {:?}", r)
        );

    }
    #[test]
    fn can_add_var_to_env(){
        let mut env = Environment::empty();
        let val = Expr::fnum(1.0);
        assert_eq!(0usize, env.num_contexts());
        env.push_context();
        assert_eq!(1usize, env.num_contexts());

        //Insert Variable
        let r = env.add_var("a", val.clone());
        assert!(r.is_ok());
        //Lookup the variable and validate
        env.lookup("a").map_or_else(
            || assert!(false, "Failed to find var in environment."),
            |(args, x)| {
                assert_eq!(val, x);
                assert_eq!(0usize, args.len());
            }
        );
        //Pop context and check variable is gone
        env.pop_context();
        env.lookup("a").map(
            |x| assert!(false, format!("Expected Err, got {:?}", x))
        );
        assert_eq!(0usize, env.num_contexts());
    }


    #[test]
    fn evaluate_symbol_in_env(){
        let val = Expr::fnum(1.0);
        let mut env = Environment::from_vars(&[
            ("a", val.clone()),
        ]);
        let s = "a";
        let sym = Expr::symbol("a");
        let r = evaluate_symbol(sym.clone(), s, &[], &mut env);
        assert_eq!(EvalResult::Expr(val.clone()), r);

    }

    #[test]
    fn evaluate_symbol_not_in_env(){
        let val = Expr::fnum(1.0); 
        let mut env = Environment::from_vars(&[
            ("a", val.clone()),
        ]);
        let s = "b";
        let sym = Expr::symbol(s);
        let r = evaluate_symbol(sym.clone(), s, &[], &mut env);
        assert_eq!(EvalResult::Expr(sym), r);
    }

    #[test]
    fn add_and_check_simple_var_in_env(){
        let expected = 5.0;
        let vals = Expr::list(&[
            Expr::symbol("let"),
            Expr::symbol("x"),
            Expr::fnum(expected),
        ]);

        let mut env = Environment::empty();
        env.push_context();
        let r = eval(vals, &mut env);
        assert_eq!(r, EvalResult::Unit);

        let lookup = Expr::symbol("x");
        let r2 = evaluate_symbol(lookup.clone(), "x", &[], &mut env);
        assert_eq!(EvalResult::Expr(Expr::fnum(expected)), r2);

        env.pop_context();
        let r3 = evaluate_symbol(lookup.clone(), "x", &[], &mut env);
        assert_eq!(EvalResult::Expr(lookup.clone()), r3);

    }

    #[test]
    fn addition_works(){
       let expr = Expr::list(&[
           Expr::symbol("+"),
           Expr::fnum(1.0),
           Expr::fnum(2.0),
           Expr::fnum(3.0),
           Expr::fnum(4.0),
           Expr::fnum(5.0),
       ]);
       let expected_sum = Expr::fnum(15.0);
       let mut env = Environment::empty();
       let r = eval(expr, &mut env);
       assert_eq!(EvalResult::Expr(expected_sum), r);

    }


    #[test]
    fn add_and_check_expr_var_in_env(){
        let expected = 15.0;
        let vals = Expr::list(&[
            Expr::symbol("let"),
            Expr::symbol("x"),
            Expr::list(&[
                Expr::symbol("+"),
                Expr::fnum(1.0),
                Expr::fnum(2.0),
                Expr::fnum(3.0),
                Expr::fnum(4.0),
                Expr::fnum(5.0),
            ]),
        ]);

        let mut env = Environment::empty();
        env.push_context();

        let r = eval(vals, &mut env);
        assert_eq!(r, EvalResult::Unit);

        let lookup = Expr::symbol("x");
        let r2 = evaluate_symbol(lookup.clone(), "x", &[], &mut env);
        assert_eq!(EvalResult::Expr(Expr::fnum(expected)), r2);

        env.pop_context();
        let r3 = evaluate_symbol(lookup.clone(), "x", &[], &mut env);
        assert_eq!(EvalResult::Expr(lookup.clone()), r3);

    }

    #[test] 
    fn add_fn_to_env(){
        let x1_sym = "x1";
        let x2_sym = "x2";
        let x1 = Expr::symbol(&x1_sym);
        let x2 = Expr::symbol(&x2_sym);
        let params = Expr::list(&[x1.clone(), x2.clone()]);
        let fn_body = Expr::list(&[Expr::symbol("+"), x1.clone(), x2.clone()]);
        let f_name = "test-func";
        let expr = Expr::list(&[
            Expr::symbol("fn"),
            Expr::symbol(&f_name),
            params.clone(),
            fn_body.clone(),
        ]);
        let mut env = Environment::empty();
        env.push_context();

        let r = eval(expr, &mut env);
        assert_eq!(r, EvalResult::Unit);

        env.lookup(&f_name).map_or_else(
            || assert!(false, "Expected function in environment but got None"),
            |(params, body)|{
                assert_eq!(&params[0], x1_sym);
                assert_eq!(&params[1], x2_sym);
                assert_eq!(body, fn_body)
            },

        );

    } 

    #[test]
    fn test_print(){
        let e1 = Expr::symbol("hello");
        let e2 = Expr::fnum(3.2);
        let e3 = Expr::list(&[
            Expr::symbol("hello"),
            Expr::symbol("world"),
        ]);

        let mut env = Environment::empty();

        assert_eq!("hello", gen_print_output(e1.clone(), &mut env));
        assert_eq!("3.2", gen_print_output(e2.clone(), &mut env));
        assert_eq!("(hello world)", gen_print_output(e3.clone(), &mut env));

    env.push_context();
       env.add_fn("test-func", &["x1".into(), "x2".into()], Expr::symbol("body")).map_or_else(
           |e| assert!(false, format!("got error {}", e)),
           |_| assert_eq!("<func-object: test-func>", gen_print_output(Expr::symbol("test-func"), &mut env)),
       );
       env.add_var("x", Expr::fnum(42.0)).map_err(|e| assert!(false, format!("got error {}", e)));
       
       let e4 = Expr::list(&[
           Expr::symbol("test-func"),
           Expr::symbol("x"),
           e3.clone(),
       ]);

       assert_eq!("(<func-object: test-func> 42 (hello world))", gen_print_output(e4.clone(), &mut env));

       let e5 = Expr::list(&[
           Expr::symbol("print"),
           e4.clone(),
           e3.clone(),
           e2.clone(),
           e1.clone(),
       ]);

       eval(e5.clone(), &mut env); 
    }

    #[test]
    fn test_default_env(){
        let  env = Environment::default();
        env.lookup("False").map_or_else(
            || assert!(false, "Expected an empty list, got None"),
            |(v, expr)| {
                assert_eq!(*expr, Expr::List(Vec::new())); 
                assert_eq!(v.len(), 0);
            });
    }

    #[test]
    fn test_eval_ite() {
        let e = Expr::list(&[
            Expr::symbol("if"),
            Expr::list(&[Expr::symbol("True")]),
            Expr::list(&[Expr::symbol("x")]),
            Expr::list(&[Expr::symbol("y")]),
        ]);

        let mut env = Environment::default();
        let result = eval(e, &mut env);
        if let EvalResult::Expr(expr) = result {
            assert_eq!(Expr::list(&[Expr::symbol("x".into())]), expr);
        } else {
            assert!(false, "Expected expression, got {:?}", result);
        }
    }

    #[test]
    fn test_ite_true_block() {
        let e = Expr::list(&[
            Expr::symbol("if"),
            Expr::list(&[Expr::symbol("True")]),
            Expr::list(&[Expr::symbol("x")]),
            Expr::list(&[Expr::symbol("y")]),
        ]);
        let mut env = Environment::default();
        let result = eval(e.clone(), &mut env);
        if let EvalResult::Expr(expr) = result {
            assert_eq!(Expr::list(&[Expr::symbol("x".into())]), expr);
        } else {
            assert!(false, "Expected expression, got {:?}", result);
        }
    }
    #[test]
    fn test_ite_false_block() {
        let e = Expr::list(&[
            Expr::symbol("if"),
            Expr::list(&[Expr::symbol("False")]),
            Expr::list(&[Expr::symbol("x")]),
            Expr::list(&[Expr::symbol("y")]),
        ]);

        let mut env = Environment::default();
        let result = eval(e, &mut env);
        if let EvalResult::Expr(expr) = result {
            assert_eq!(Expr::list(&[Expr::symbol("y".into())]), expr);
        } else {
            assert!(false, "Expected expression, got {:?}", result);
        }
    }     
    #[test]
    fn test_func_evaluation() {
        let x = 2.0;
        let x_name = Expr::symbol("x");
        let y = 3.0;
        let y_name = Expr::symbol("y");

        let fn_name = Expr::symbol("test-func");

        // (fn test-func (x y) (+ x y))
        let fn_def = Expr::list(&[
            Expr::symbol("fn"),
            fn_name.clone(),
            Expr::list(&[x_name.clone(), y_name.clone()]),
            Expr::list(&[Expr::symbol("+"), x_name.clone(), y_name.clone()]),
        ]);

        let mut env = Environment::empty();
        env.push_context();
        let r = eval(fn_def, &mut env);
        assert_eq!(EvalResult::Unit, r);

        // (test-func (+ 2 3) 1) -> Expect Expr:FNum(6.0)
        let fn_eval = Expr::list(&[
            fn_name.clone(),
            Expr::list(&[Expr::symbol("+"), Expr::fnum(x), Expr::fnum(y)]),
            Expr::fnum(1.0),
        ]);
        let r2 = eval(fn_eval, &mut env);
        if let EvalResult::Expr(e) = r2 {
            if let Expr::FNum(n) = *e {
                assert_eq!(n, x + y + 1.0);
            } else {
                assert!(false, format!("Expected FNum(6.0), got {:?}", e));
            }
        } else {
            assert!(false, format!("Expected Expr::fnum(6.0), got {:?}", r2));
        }
    }
}
