((let x 5)
(fn func-1 (x y) (+ x y))
(print creating y)
(let y (if (func-1 x 4) (hello) (world)))
(print printing results)
(print x)
(print y))
