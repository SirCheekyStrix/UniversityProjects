import itertools
import string
import time

def brute_force(target):
    characters = string.ascii_letters + string.digits  # Dostępne znaki (małe, wielkie litery i cyfry)
    attempt_count = 0
    
    print(f"Rozpoczynanie łamania ciągu: {target}")
    
    start_time = time.time()
    
    for length in range(1, len(target) + 1):  # Długość kombinacji
        for attempt in itertools.product(characters, repeat=length):
            attempt_count += 1
            attempt = ''.join(attempt)
            
            if attempt == target:
                end_time = time.time()
                print(f"Znaleziono pasujący ciąg: {attempt}")
                print(f"Liczba prób: {attempt_count}")
                print(f"Czas wykonania: {end_time - start_time:.2f} sekundy")
                return
            
            # Opcjonalne logowanie (może być wolniejsze):
            # print(f"Próba: {attempt}")

    print("Nie udało się znaleźć pasującego ciągu.")

# Przykład użycia
brute_force("GhyZer0202")
