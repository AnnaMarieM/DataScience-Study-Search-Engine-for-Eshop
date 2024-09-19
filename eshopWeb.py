import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import modulu CORS
import stanza

# Český model pro Stanza
stanza.download('cs')
nlp = stanza.Pipeline('cs')

app = Flask(__name__)
CORS(app)  # Povolit CORS pro všechny trasy

# Vlastní seznam stopwords pro češtinu
czech_stopwords = set([
    "a", "aby", "aj", "ale", "anebo", "ani", "ano", "asi", "až", "během", "bez", "by", "byl", "byla", "byli", "bylo", "byly", 
    "být", "co", "což", "cz", "či", "dále", "další", "den", "deset", "devět", "do", "dobrý", "docela", "dva", "dvě", "ho", 
    "jak", "jakmile", "jako", "je", "jeho", "její", "jejich", "jemu", "jen", "jenom", "ještě", "jestli", "jestliže", "ji", 
    "jich", "jím", "jimi", "jinak", "jsem", "jsme", "jsi", "jsou", "jste", "já", "kam", "kde", "kdo", "kdy", "když", "ke", 
    "kdo", "kdy", "když", "konec", "která", "které", "kterou", "který", "kteří", "když", "kvůli", "má", "mají", "málo", 
    "máme", "máš", "mezi", "mi", "mít", "mně", "mnou", "moc", "mohl", "mohou", "moje", "moji", "můj", "musí", "muž", "my", 
    "na", "nad", "nám", "námi", "naproti", "naše", "naši", "ne", "nebo", "nebyl", "nebyla", "nebyli", "nebyly", "něco", 
    "nedělá", "nedělají", "nedělám", "neděláme", "neděláš", "neděláte", "nějak", "nějaký", "nejsi", "nejsou", "nemají", 
    "nemáme", "nemáš", "nemáte", "není", "nestačí", "než", "nic", "nich", "ním", "nimi", "nové", "nový", "nových", "novým", 
    "novými", "nyní", "o", "od", "ode", "on", "ona", "oni", "ono", "ony", "osm", "pak", "patnáct", "pět", "po", "pořád", 
    "potom", "pouze", "právě", "pro", "proč", "proto", "protože", "první", "před", "přes", "přese", "při", "přičemž", "s", 
    "se", "si", "skoro", "smějí", "smí", "snad", "spolu", "sta", "sté", "sto", "strana", "sté", "své", "svých", "svým", 
    "svými", "ta", "tady", "tak", "také", "takže", "tamhle", "tamhleto", "tamto", "tě", "tebe", "tebou", "ted", "tedy", 
    "ten", "tento", "tím", "tímto", "tito", "to", "tohle", "toho", "tohoto", "tom", "tomto", "tomu", "tomuto", "tu", 
    "tuto", "ty", "tyto", "u", "už", "v", "vám", "vámi", "vás", "vaše", "vaši", "ve", "vedle", "vlastně", "vy", "však", 
    "všechen", "všichni", "všichno", "za", "zatímco", "ze", "že"
])

# Data
data = {
    "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "název": [
        "Televize Samsung LED 155 palců",
        "Televize Sony LED 55 palců",
        "Notebook Lenovo ThinkPad",
        "Notebook MacAir",
        "Notebook MacPro",
        "Chytrý telefon Samsung Galaxy S20",
        "Chytrý telefon iPhone 15",
        "Bezdrátová sluchátka Sony",
        "Bezdrátová sluchátka Pioneer",
        "Sluchátka Pioneer"
    ],
    "popis": [
        "155-palcová televize s LED technologií a rozlišením 7K Ultra HD. Skvělá volba pro domácí zábavu.",
        "55-palcová televize s LED technologií a rozlišením 4K Ultra HD. Skvělá volba pro domácí zábavu.",
        "Tenký notebook s výkonným procesorem Intel Core i7 a operační pamětí 16 GB. Ideální pro profesionální využití.",
        "Tenký notebook s výkonným procesorem Intel Core i7 a operační pamětí 16 GB. Ideální pro domácí využití.",
        "Notebook s výkonným procesorem Intel M2 a operační pamětí 32 GB. Ideální pro profesionální využití.",
        "Chytrý telefon s výkonným procesorem, špičkovým fotoaparátem a podporou 5G sítí. Ideální pro multitasking a mobilní fotografování.",
        "Chytrý telefon s výkonným procesorem, špičkovým fotoaparátem a podporou 5G sítí. Ideální pro multitasking a mobilní fotografování.",
        "Bezdrátová sluchátka s výborným zvukem a dlouhou výdrží baterie. Ideální pro poslech hudby a hovory bez omezení kabelů.",
        "Bezdrátová sluchátka s výborným zvukem a dlouhou výdrží baterie. Ideální pro poslech hudby a hovory bez omezení kabelů.",
        "Bezdrátová sluchátka s výborným zvukem a připojená kabelem. Ideální pro poslech hudby."
    ],
    "kategorie": [
        "Televize", "Televize", "Notebook", "Notebook", "Notebook", "Chytrý telefon", "Chytrý telefon", "Sluchátka", "Sluchátka", "Sluchátka"
    ],
    "cena": [
        1899.59, 799.99, 1299.99, 2277.88, 4277.88, 999.99, 1099.99, 129.88, 229.99, 409.21
    ],
    "recenze": [
        "recenze 4 z 10", "recenze 8 z 10", "recenze 4 z 10", "recenze 4 z 10", "recenze 10 z 10", "recenze 4 z 10", "recenze 7 z 10", "recenze 4 z 10", "recenze 4 z 10", "recenze 9 z 10"
    ]
}

# Načtení dat do DataFrame
products = pd.DataFrame(data)
products["popisujici_slova"] = products["název"] + " " + products["popis"] + " " + products["kategorie"] + " " + products["kategorie"] + " " + products["kategorie"] + " " + products["kategorie"]


# Funkce pro lemmatizaci a odstranění stopwords
def lemmatize_and_remove_stopwords(text):
    doc = nlp(text)
    lemmas = [word.lemma for sentence in doc.sentences for word in sentence.words if word.lemma not in czech_stopwords]
    return " ".join(lemmas)


# Aplikace funkce na sloupec "popisujici_slova"
products["popisujici_slova"] = products["popisujici_slova"].apply(lemmatize_and_remove_stopwords)


# Vytvoření korpusu popisů produktů
korpus = products["popisujici_slova"].tolist()

# Výpočet TF-IDF matice pro produkty
Vector_Space_model = TfidfVectorizer()
tfidf_matrix = Vector_Space_model.fit_transform(korpus)

@app.route("/", methods=["POST"])
def query_products():
    # Získat JSON data z POST žádosti
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Dotaz nebyl poskytnut"}), 400

    query = data["query"]
    query = lemmatize_and_remove_stopwords(query)


    # Vytvoření vektoru dotazu pomocí stejného TF-IDF vectorizeru
    query_vector = Vector_Space_model.transform([query])

    # Výpočet kosinové podobnosti mezi vektorem dotazu a vektory produktů
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Získání a setrizeni relevantních produktových ID a jejich skóre (výsledky, jež nemají nulové score)
    relevantni_produkty = {
        produkt_id: score
        for produkt_id, score in zip(products["id"], cosine_similarities)
        if score > 0
    }

    serazene_produkty = sorted(
        relevantni_produkty.items(), key=lambda item: item[1], reverse=True
    )

    # Připravit výsledky
    results = []
    for produkt_id, score in serazene_produkty:
        produkt = products.loc[products["id"] == produkt_id].iloc[0]
        result = {
            "id": produkt_id,
            "názov": produkt["název"],
            "popis": produkt["popis"],
            "cena": produkt["cena"],
            "recenzie": produkt["recenze"],
            "score": score
        }
        results.append(result)

    print(results)

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="localhost", port=8080)
