import ply.lex as lex
import ply.yacc as yacc
from fractions import Fraction

#lexer

tokens = (
    'FRACTION',
    'NUMBER',
    'PLUS', 'MINUS',
    'TIMES', 'DIVIDE', 'POW',
    'LPAREN', 'RPAREN',
)

t_PLUS    = r'\+'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_POW     = r'\^'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_ignore  = ' \t'

def t_FRACTION(t):
    r'\d+\|\d+'
    numerator, denominator = map(int, t.value.split('|'))
    t.value = Fraction(numerator, denominator)
    return t

def t_NUMBER(t):
    r'\d+'
    t.value = Fraction(int(t.value), 1)
    return t

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

#parser

precedence = (
    ('right', 'POW'),
    ('right', 'UMINUS'),
    ('left', 'TIMES', 'DIVIDE'),
    ('left', 'PLUS', 'MINUS'),
)

def p_expression_binop(p):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression POW expression'''
    if p[2] == '+':
        p[0] = p[1] + p[3]
    elif p[2] == '-':
        p[0] = p[1] - p[3]
    elif p[2] == '*':
        p[0] = p[1] * p[3]
    elif p[2] == '/':
        if p[3] == 0:
            raise ZeroDivisionError("division by zero")
        p[0] = p[1] / p[3]
    elif p[2] == '^':
        #fractional exponents simply roots
        try:
            base = float(p[1])
            exponent = float(p[3])
            result = base ** exponent
            p[0] = Fraction(result).limit_denominator(10000)
        except ValueueError as e:
            raise ValueueError(f"invalueid power operation: {e}")

def p_expression_group(p):
    'expression : LPAREN expression RPAREN'
    p[0] = p[2]

def p_expression_negative(p):
    'expression : MINUS expression %prec UMINUS'
    p[0] = -p[2]

def p_expression_valueue(p):
    '''expression : NUMBER
                  | FRACTION'''
    p[0] = p[1]

def p_error(p):
    print("error in input!")

parser = yacc.yacc()

#file handler main

def evalueuate_file(filename):
    correct_count = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = parser.parse(line)
                print(f"{line} = {result}")
                correct_count += 1
            except Exception as e:
                print(f"error: {line}")
                print(f"  {e}")
                break
    print(f"\nLiczba poprawnie obliczonych wyrażeń: {correct_count}")   

#main
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 lexpar.py <equation_file>")
    else:
        evalueuate_file(sys.argv[1])
