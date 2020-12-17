use crate::lex::Token;
use crate::types::Expr;
use std::rc::Rc;

#[derive(Debug)]
pub enum ParseError {
    BadParse(String),
    EOF,
}

#[derive(Debug)]
pub enum ParseResult {
    Success(usize, Rc<Expr>),
    Failure(ParseError),
}

pub fn parse(tokens: &[Token]) -> Result<Rc<Expr>, ParseError> {
    match parser(tokens, 0){
        ParseResult::Success(_, expr) => Ok(expr),
        ParseResult::Failure(err) => Err(err),

    }
}

fn parser(tokens: &[Token], index: usize) -> ParseResult{
    let mut index = index;
    if let Some(mut t) = tokens.get(index){
        match &*t {
            Token::LPar => {
                index+=1;
                let mut exprs = Vec::new();
                while  *t != Token::RPar{
                    match parser(tokens, index){
                        ParseResult::Success(idx, expr) => {
                            exprs.push(expr);
                            index = idx;
                        },
                        e => return e,
                    }
                    if index >= tokens.len() {
                        return ParseResult::Failure(ParseError::BadParse(
                            "Unclosed delimitor".into(),
                        ))
                    }
                    t = &tokens[index];
                }
                ParseResult::Success(index + 1, Expr::list(&exprs))
            },
            Token::RPar => {
                ParseResult::Failure(ParseError::BadParse("Unexpected ) encountered.".into()))
            },
            Token::Literal(s) => {
                if let Ok(n) = s.parse::<f64>(){
                    ParseResult::Success(index + 1, Expr::fnum(n))

                }
                else{
                    ParseResult::Success(index + 1, Expr::symbol(&s))
                }

            },
            _ => ParseResult::Failure(ParseError::BadParse(format!("unknown token: {:?}", *t))),
        }
    }
    else{
        ParseResult::Failure(ParseError::EOF)
    }
}



#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_fnum(){
        let res = parser(&[Token::Literal("1".into())], 0);
        if let ParseResult::Success(i, expr) = res {
            assert_eq!(i, 1);
            if let Expr::FNum(n) = *expr {
                assert_eq!(n, 1.0);
            }
            else{
                assert!(false, format!("expected fnum, got {:?}", *expr)); 
            }
        }else{
            assert!(false, format!("Expected success, failed with {:?}", res));
        }
    }
    #[test]
    fn parse_symbol(){
        let res = parser(&[Token::Literal("hello".into())], 0);
        if let ParseResult::Success(i, expr) = res {
            assert_eq!(i, 1);
            if let Expr::Symbol(s) = &*expr {
                assert_eq!(s, "hello");
            }
            else{
                assert!(false, format!("expected Symbol, got {:?}", *expr)); 
            }
        }else{
            assert!(false, format!("Expected success, failed with {:?}", res));
        }
    }

    #[test]
    fn parse_list(){
        let tokens = [
            Token::LPar,
            Token::Literal("+".into()),
            Token::Literal("1.8".into()),
            Token::Literal("1.2".into()),
            Token::RPar
        ];
        let expected = Expr::list(&[
            Expr::symbol("+"),
            Expr::fnum(1.8),
            Expr::fnum(1.2),
        ]);
        let res =  parser(&tokens, 0);
        if let ParseResult::Success(i, expr) = res {
            assert_eq!(i, tokens.len());
            assert_eq!(expr, expected);
        }
        else{
            assert!(false, format!("Expected success, failed with {:?}", res));
        }
    }

    #[test]
    fn parse_nested_symbol(){
        let tokens = [
            Token::LPar,
            Token::LPar,
            Token::Literal("A".into()),
            Token::RPar,
            Token::RPar,
        ];
        let expected = Expr::list(&[Expr::list(&[Expr::symbol(("A"))])]);
        let res = parser(&tokens, 0);
        if let ParseResult::Success(i, expr) = res {
            assert_eq!(i, tokens.len());
            assert_eq!(expected, expr);
        }
        else{ 
            assert!(false, format!("Expected Success, failed with {:?}", res));
        }
    }

    #[test]
    fn nested_lists(){
        let tokens = [
            Token::LPar,
            Token::Literal("+".into()),
            Token::LPar,
            Token::Literal("+".into()),
            Token::Literal("1.2".into()),
            Token::Literal("-12.8".into()),
            Token::RPar,
            Token::LPar,
            Token::Literal("*".into()),
            Token::Literal("1.2".into()),
            Token::Literal("-12.8".into()),
            Token::RPar,
            Token::RPar,
        ];
        let expected = Expr::list(&[
            Expr::symbol("+"),
            Expr::list(&[Expr::symbol("+"), Expr::fnum(1.2), Expr::fnum(-12.8)]),
            Expr::list(&[Expr::symbol("*"), Expr::fnum(1.2), Expr::fnum(-12.8)]),
        ]);
        let res = parser(&tokens, 0);
        if let ParseResult::Success(i, expr) = res {
            assert_eq!(i, tokens.len());
            assert_eq!(expected, expr);
        }
        else{
            assert!(false, format!("Expected Success, failed with {:?}", res));
        }
    }
}
