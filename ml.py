#!/usr/bin/env python3
"""
Prosty interpreter (dekoder) maszyny licznikowej z instrukcjami:
    Z(x)       - wyzeruj rejestr x
    S(x)       - zwiększ rejestr x o 1
    T(x,y)     - skopiuj zawartość rejestru x do rejestru y
    I(x,y,t)   - jeśli rejestr x == rejestr y to skocz do instrukcji t (1-based)

Format programu (plik tekstowy lub stdin): każda instrukcja w nowej linii,
można użyć formatu z nawiasami lub spacjami, np.:
    Z(1)
    S 1
    T(1,2)
    I 1 2 4

Uruchomienie:
    python3 register_machine.py program.txt
Jeśli nie podasz pliku, program zostanie wczytany ze stdin.
"""

import sys
import re

INSTR_RE = re.compile(r'^\s*([ZS T I])\s*(?:\(\s*([0-9]+)(?:\s*,\s*([0-9]+))?(?:\s*,\s*([0-9]+))?\s*\)|\s+([0-9]+)(?:\s+([0-9]+))?(?:\s+([0-9]+))?)\s*(?:#.*)?$'.replace(' ', ''))

def parse_line(line):
        m = INSTR_RE.match(line)
        if not m:
                raise ValueError(f'Nieprawidłowa instrukcja: {line!r}')
        op = m.group(1)
        # groups 2-4 are from parentheses, 5-7 from space-separated
        a = m.group(2) or m.group(5)
        b = m.group(3) or m.group(6)
        c = m.group(4) or m.group(7)
        nums = [int(x) for x in (a, b, c) if x is not None]
        if op == 'Z' or op == 'S':
                if len(nums) != 1:
                        raise ValueError(f'Instrukcja {op} wymaga 1 argumentu: {line!r}')
                return (op, nums[0])
        if op == 'T':
                if len(nums) != 2:
                        raise ValueError(f'Instrukcja T wymaga 2 argumentów: {line!r}')
                return (op, nums[0], nums[1])
        if op == 'I':
                if len(nums) != 3:
                        raise ValueError(f'Instrukcja I wymaga 3 argumentów: {line!r}')
                return (op, nums[0], nums[1], nums[2])
        raise ValueError('Nieznana instrukcja')

def load_program(lines):
        prog = []
        for raw in lines:
                line = raw.strip()
                if not line or line.startswith('#'):
                        continue
                prog.append(parse_line(line))
        return prog

def ensure_registers(regs, idx):
        if idx < 1:
                raise IndexError('Indeksy rejestrów zaczynają się od 1')
        while len(regs) < idx:
                regs.append(0)

def run_program(prog, regs=None, max_steps=100000, trace=False):
        if regs is None:
                regs = []
        regs = list(regs)[:]  # copy
        ip = 1  # instrukcja 1-based
        steps = 0
        history = []
        n = len(prog)
        while 1 <= ip <= n:
                if steps >= max_steps:
                        raise RuntimeError(f'Osiągnięto limit kroków ({max_steps})')
                instr = prog[ip-1]
                if trace:
                        history.append((steps, ip, instr, list(regs)))
                op = instr[0]
                if op == 'Z':
                        x = instr[1]
                        ensure_registers(regs, x)
                        regs[x-1] = 0
                        ip += 1
                elif op == 'S':
                        x = instr[1]
                        ensure_registers(regs, x)
                        regs[x-1] += 1
                        ip += 1
                elif op == 'T':
                        x, y = instr[1], instr[2]
                        ensure_registers(regs, max(x, y))
                        regs[y-1] = regs[x-1]
                        ip += 1
                elif op == 'I':
                        x, y, t = instr[1], instr[2], instr[3]
                        ensure_registers(regs, max(x, y))
                        if regs[x-1] == regs[y-1]:
                                if not (1 <= t <= n):
                                        raise IndexError(f'Skok do nieistniejącej instrukcji: {t}')
                                ip = t
                        else:
                                ip += 1
                else:
                        raise RuntimeError('Nieznana instrukcja w czasie wykonania')
                steps += 1
        return regs, history

def print_trace(history):
        for step, ip, instr, regs in history:
                print(f'{step:5d}: ip={ip} instr={instr} regs={regs}')

def main():
        if len(sys.argv) >= 2:
                with open(sys.argv[1], 'r', encoding='utf-8') as f:
                        lines = f.readlines()
        else:
                print('Wczytywanie programu ze stdin. Zakończ Ctrl+D (Unix) lub Ctrl+Z (Windows).')
                lines = sys.stdin.readlines()

        prog = load_program(lines)
        # przykładowe wartości początkowe rejestrów: użytkownik może zmienić tu
        initial_regs = []
        # proste wczytanie wartości początkowych z argumentu opcjonalnego
        if len(sys.argv) >= 3:
                # np. python3 script.py program.txt "1,0,5"
                try:
                        initial_regs = [int(x) for x in sys.argv[2].split(',') if x.strip()!='']
                except Exception:
                        print('Niepoprawny format początkowych rejestrów. Oczekiwano: "a,b,c"')
                        sys.exit(1)

        try:
                regs, history = run_program(prog, regs=initial_regs, max_steps=100000, trace=True)
        except Exception as e:
                print('Błąd wykonania:', e)
                sys.exit(1)

        print('--- Trace ---')
        print_trace(history)
        print('--- Wynik końcowy ---')
        print('Rejestry (1-based):', regs)

if __name__ == '__main__':
        main()