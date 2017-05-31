# inf5860_uke1

Hent koden til oppgavene med:

    git clone https://github.com/sigmunjr/inf5860_uke1.git

I filene *indexing.py*, *math_operations.py* og *konvolusjon.py*, er det oppgaver for å lære nettopp indeksering, matematiske
operasjoner og konvolusjoner ved hjelp av *python* og *numpy*.
I hver av disse filene er det en rekke funksjoner f.eks. math1, math2 osv. Din oppgave er å få disse funksjonene til å returnere
riktig resultat. Oppgave teksten står under definisjonen av funksjonen. f.eks.

```python
def math1(mat):
  """
  Square each value in mat separatly
  """
  return mat
```

Denne oppgaven løses ved å endre funksjonen slik:

```python
def math1(mat):
  """
  Square each value in mat separatly
  """
  return mat**2
```

For å teste om du har løst oppgavene riktig kan du kjøre:

    $ nosetests

Fra mappen inf5860_uke1. Da blir filene med test_ kjørt og man får et svar på hvor mange oppgaver som er gjort riktig.
For å teste en enkelt oppgave kan du f.eks. kjøre (her testes funksjonen index1 i filen indexing.py):

    $ nosetests test_indexing.py:test_index1

Husk at du også kan lese test koden i f.eks. *test_indexing.py* for å bedre forstå oppgaven.

**Løsningene finner du ved**

    $ git clone https://github.com/sigmunjr/inf5860_uke1_solutions.git
