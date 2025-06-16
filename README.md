# py_proteins

## How to run the project

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment (Windows)
.\venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Run the application
python main.py
```

**For Linux/Mac:**
```bash
source venv/bin/activate  # Step 2 alternative
```


## **PYTANIA**

### **1. Punktacja sekwencji białek**
🔹 W podesłanym przykładowym projekcie przyznawane są punkty sekwencjom merów, a konkretne sekwencje (np. `TAT`) otrzymują dodatkowe punkty.    
- W jaki sposób należy przyznawać punkty sekwencjom białek?  
- Co powinno być wyróżnikiem otrzymującym dodatkowe punkty?  

---

### **2. Długość sekwencji białek**
🔹 W danych dotyczących białek z zaznaczonymi domenami występuje sekwencja o długości ponad **2400 znaków**.  
- Czy dla dalszej analizy potrzebujemy, aby wszystkie sekwencje miały tę samą długość?  
- Czy tablice reprezentujące sekwencje powinny mieć długość **2444** (dopasowane do najdłuższej sekwencji)?  

---

### **3. Problem z funkcją `get_negative()` i danymi z UniProt**  
🔹 W funkcji `get_negative()` pobieram dane za pomocą **API UniProt**.  
- Filtry na stronie UniProt pokazują **5075 wyników**, ale funkcja zwraca tylko **5065 białek**.  
- Dodatkowo, filtr ograniczający długość sekwencji do **200** nie działa poprawnie — w wynikach pojawia się białko o długości **385**.   
- Czy to jest znany błąd API UniProt, czy może błąd w implementacji funkcji?   

---
