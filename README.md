# Classeur de documents

Outil local de catégorisation automatique de documents (PDF, TXT, MD, CSV, TSV, LOG, DOCX), **sans LLM et sans connexion internet après l'installation**. Chaque utilisateur entraîne son propre modèle sur ses propres documents, réutilisable indéfiniment hors-ligne.

## Pourquoi cet outil

- **Pas de LLM, pas de coûts d'API** : la catégorisation repose sur des techniques classiques de machine learning (TF-IDF ou embeddings de phrases + clustering/classification), pas sur un modèle de langage.
- **100% local** : vos documents (factures, contrats, relevés fiscaux...) ne quittent jamais votre machine.
- **Un modèle par utilisateur** : le modèle entraîné (`model.pkl`) est propre à vos données et à vos catégories — il ne dépend d'aucun service externe pour fonctionner.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

Paquets optionnels, à installer seulement si besoin :

| Fichier | Active | Coût |
|---|---|---|
| `requirements-embeddings.txt` | le moteur d'analyse "embeddings" (sémantique) | téléchargement d'un modèle ~90 Mo au premier lancement |
| `requirements-gui.txt` | le vrai glisser-déposer dans l'interface graphique | paquet léger (`tkinterdnd2`) |
| `requirements-docx.txt` | la lecture des fichiers `.docx` (Word) | paquet léger (`python-docx`) |
| `requirements-office.txt` | la lecture des fichiers `.xlsx`/`.xlsm` (Excel) et `.pptx` (PowerPoint) | paquets légers (`openpyxl`, `python-pptx`) |
| `requirements-msg.txt` | la lecture des emails Outlook `.msg` | paquet léger (`extract-msg`) |

## Lancer l'interface graphique

```bash
python ml_pdf_gui.py
```

La fenêtre s'ouvre sur cinq onglets, dans cet ordre : **Classification** (à gauche — c'est l'usage le plus fréquent), puis **Entraînement**, **Transformer les données**, **Automatisation** et **Paramètres**, tous regroupés à sa droite (le flux de mise en place et de configuration, dans l'ordre où on s'en sert).

```
┌───────────────┐  produit   ┌───────────┐  renomme les   ┌─────────────────────┐  réutilisé par  ┌────────────────────────┐
│  Entraînement │ ─────────► │ model.pkl │ ─────────────► │ Transformer les      │ ───────────────►│ Classification /        │
│  (une fois)   │            │ (figé)    │  catégories    │ données (optionnel)  │                  │ Automatisation (souvent)│
└───────────────┘            └───────────┘                └─────────────────────┘                  └────────────────────────┘
```

---

## Étape 1 — Entraînement (créer le modèle)

Aucun tri manuel n'est nécessaire : on pointe un dossier de documents **en vrac**, l'outil détecte les catégories tout seul par similarité entre documents (les factures se regroupent avec les factures, les contrats avec les contrats, etc., sans qu'on ait besoin de le lui dire à l'avance).

**Cas d'usage** — en haut de l'onglet, un menu déroulant propose des points de départ pour des besoins courants : sélectionner l'un d'eux pré-remplit les types de fichiers et le moteur adaptés.

| Cas d'usage | Statut | Détail |
|---|---|---|
| Classeur de documents | Disponible | Mode par défaut de l'onglet (tous types de fichiers, TF-IDF) |
| Trieur d'emails | Disponible | Limite aux formats `.eml`/`.msg`, embeddings multilingues — renommez ensuite les catégories détectées dans Transformer les données pour imposer des catégories fixes (ex. spam / non-spam) |
| Extracteur de factures | À venir | Nécessiterait un modèle d'extraction de champs (NER) entraîné sur des factures annotées (ex. SROIE, CORD) — capacité différente de la catégorisation, pas encore implémentée |
| Analyseur de contrats | À venir | Nécessiterait un modèle de détection de clauses (type BERT finement ajusté sur un jeu de données comme CUAD) — pas encore implémenté |
| Détecteur de doublons / plagiat | À venir | Nécessiterait une recherche de similarité par embeddings à l'échelle d'une base documentaire entière — pas encore implémenté |

1. **Choisir le dossier de documents à analyser** — un seul dossier contenant tous vos fichiers en vrac (`Parcourir...`). La case **"Inclure les sous-dossiers"** (cochée par défaut) fait aussi remonter les documents rangés dans des sous-dossiers de ce dossier — y compris, volontairement, le dossier `dataset/` d'un autre modèle si vous l'utilisez comme dossier source (chaînage de modèles). Les dossiers `_backup` (doublons mis de côté, voir plus bas), `pkl_history`, `json_history` et `dataset_history` sont toujours ignorés lors d'un scan récursif, pour ne pas ré-analyser indéfiniment les mêmes fichiers mis de côté ou archivés. Ce même réglage (récursif ou non) est mémorisé dans le modèle et repris automatiquement par "Améliorer le modèle" (onglet Classification) et par un nouveau ré-entraînement basé sur ce modèle.
2. **Choisir les types de fichiers à inclure** — cases à cocher groupées par famille (Documents, Texte et données, Balisage, Email, Bureautique). Toutes cochées par défaut (tous les formats pris en charge) ; décochez pour restreindre à un seul type, plusieurs, ou tout autre sous-ensemble. Utile par exemple pour n'entraîner que sur des emails, ou exclure des formats de données structurées (`.json`, `.csv`...) qui ne sont pas de vrais "documents".
3. **"2. Moteur d'analyse"** — un menu de **préréglages** fixe d'un coup le moteur et tous ses paramètres pour CET entraînement précis (sans toucher aux valeurs par défaut de Paramètres) ; chaque champ reste ensuite modifiable individuellement pour affiner. Le bouton **"? Explications"** ouvre une fenêtre dédiée avec le détail complet de chaque moteur, de chaque modèle d'embeddings et de chaque préréglage — rien n'encombre le formulaire par défaut.

   | Préréglage | Moteur | Quand l'utiliser |
   |---|---|---|
   | Par défaut (valeurs de Paramètres) | TF-IDF | Aucun réglage particulier, reprend l'onglet Paramètres |
   | Documents très gabarités (même modèle) | TF-IDF | Factures d'un même fournisseur, fiches de paie d'une même personne — peu de catégories attendues |
   | Mélange de plusieurs types de documents | TF-IDF | Factures + contrats + fiches de paie mélangés — vocabulaire élargi |
   | Gros volume de documents, priorité à la vitesse | TF-IDF | Vocabulaire réduit, mots seuls — rapide sur un très grand nombre de documents |
   | Exploration permissive | TF-IDF | Seuils très bas pour tout faire ressortir, à trier ensuite dans Transformer les données |
   | Texte libre, léger et rapide (anglais) | Embeddings `all-MiniLM-L6-v2` | Premier essai en embeddings, sans attendre |
   | Texte libre, un peu plus précis (anglais) | Embeddings `all-MiniLM-L12-v2` | Un peu plus précis que L6, reste rapide |
   | Texte libre en français, formulations variées | Embeddings `paraphrase-multilingual-MiniLM-L12-v2` | Contrats variés, courriers, rapports en français |
   | Texte libre, meilleure qualité (anglais) | Embeddings `all-mpnet-base-v2` | Meilleure qualité en anglais, plus lent |
   | Texte libre multilingue, meilleure qualité (lourd) | Embeddings `paraphrase-multilingual-mpnet-base-v2` | La meilleure qualité multilingue disponible, le plus lent |

   Les modèles d'embeddings se téléchargent une seule fois (~90 Mo à ~970 Mo selon le modèle) puis fonctionnent hors-ligne.

   **TF-IDF ou embeddings : lequel choisir ?** Mesuré sur un vrai jeu de test (documents très gabarités, même mise en page et vocabulaire quasi identique d'un exemplaire à l'autre — factures d'un même fournisseur, fiches de paie, relevés d'un même organisme...) : dans ce cas, les embeddings n'apportent **aucun gain**, y compris un modèle multilingue sur des documents en français — ce qui varie d'un exemplaire à l'autre y est numérique (dates, montants), pas sémantique, donc la compréhension du sens n'aide pas. **TF-IDF reste le meilleur choix par défaut** sur ce type de documents administratifs très gabarités : gratuit, instantané, aucun téléchargement. Réservez les embeddings — et en particulier les modèles multilingues, plus lourds — à un corpus **réellement hétérogène en texte libre** (contrats de nature variée, courriers, rapports), où deux documents d'une même catégorie peuvent être formulés très différemment sans partager le même vocabulaire exact. Détail complet dans la fenêtre "? Explications".

   **Important — le score qui décide s'il y a "assez" de catégories n'a pas la même échelle selon le moteur.** TF-IDF opère dans un espace de mots très creux et de grande dimension où le score de silhouette (qui mesure la qualité d'un découpage) reste structurellement bas — mesuré sur un corpus de test à 3 catégories réellement distinctes : ~0.03 pour TF-IDF contre ~0.19 pour les embeddings, sur le **même découpage correct**. Il n'existe donc pas de seuil unique parfait pour les deux moteurs : un réglage assez permissif pour laisser TF-IDF détecter de vraies catégories laissera parfois passer, sur un dossier réellement homogène, un découpage un peu trop fin — c'est une limite connue, pas un bug, et l'onglet Transformer les données reste le moyen le plus fiable de corriger ce cas ponctuellement (fusionner des catégories en trop).

   Deux cases à cocher complètent les paramètres avancés :
   - **Noms de catégorie plus naturels (KeyBERT)** (cochée par défaut) — au lieu d'assembler simplement les mots les mieux pondérés en TF-IDF, [KeyBERT](https://github.com/MaartenGr/KeyBERT) choisit les mots/groupes de mots les plus proches sémantiquement du sens global de la catégorie (via le modèle d'embeddings `all-MiniLM-L6-v2`, déjà utilisé ailleurs dans l'outil — aucun téléchargement supplémentaire), avec diversification pour éviter des variantes d'un même mot. Se replie automatiquement sur le nommage TF-IDF si le paquet optionnel `keybert` n'est pas installé (`pip install -r requirements-embeddings.txt`).
   - **Détecter les documents en double** (cochée par défaut) — à partir des mêmes vecteurs déjà calculés pour le regroupement (aucun calcul supplémentaire lourd), signale les paires de documents quasi identiques (similarité ≥ 97 %) dans le journal et dans le résumé du corpus (voir plus bas). Désactivé automatiquement au-delà de 4000 documents pour éviter un calcul trop long. Si des doublons sont trouvés, une fenêtre les liste à la fin de l'entraînement et propose de les déplacer vers un dossier **`_backup`** créé à côté de leur dossier d'origine — jamais une suppression définitive, toujours récupérables au besoin. Un exemplaire de chaque groupe de quasi-doublons est toujours conservé.
4. **(Optionnel) Choisir un modèle existant à améliorer** — au lieu de repartir de zéro, on peut sélectionner un `model.pkl` déjà créé : le nouveau modèle sera alors entraîné sur l'ancien jeu de documents **et** les nouveaux, ce qui affine les catégories au fil du temps sans perdre ce qui a déjà été appris. Les **renommages et suppressions faits dans "Transformer les données" sont conservés** : chaque document repasse d'abord dans l'ancien modèle pour connaître son ancienne catégorie (nouveau nom ou "autre" en cas de suppression), et cette catégorie est reprise pour toute nouvelle catégorie qui lui correspond majoritairement — seuls les sujets vraiment nouveaux reçoivent un nom fraîchement généré.
5. **Choisir le nom du modèle** (juste un nom, pas un chemin — l'emplacement se déduit automatiquement, affiché en direct sous le champ) et cliquer sur **"Entraîner le modèle"**.

Pendant le traitement, une barre de progression s'anime et le journal affiche en direct chaque étape (extraction du texte document par document, vectorisation, recherche du nombre de catégories) — pour ne jamais donner l'impression que l'outil est figé, même sur un gros dossier.

Le journal en bas de l'onglet affiche les catégories détectées (ex. `factures_acme_tva`, `bourse_degiro_rapport`...) et confirme l'enregistrement du modèle.

### Où sont rangés les modèles

Chaque modèle nommé `<nom>` obtient son propre dossier, **toujours au même endroit** — retrouvable même après avoir fermé et rouvert l'application, sans dépendre d'un dossier temporaire :

```
storage/models/<nom>/
    <nom>.pkl              le modèle entraîné
    <nom>.json               les catégories et les fichiers qui y contribuent
    <nom>_digest.json        résumé condensé du corpus (voir ci-dessous)
    dataset/                   une COPIE de chaque document utilisé, organisée par catégorie
      <catégorie>/               (jamais les documents d'origine, qui ne sont jamais déplacés)
    pkl_history/, json_history/, dataset_history/
                                  instantanés automatiques avant chaque modification (voir plus bas)
```

Le dossier `dataset/` est **cumulatif** : chaque entraînement, amélioration depuis l'onglet Entraînement ou depuis la Classification y ajoute ses documents (ou les déplace si leur catégorie change), sans jamais perdre ce qui a été ajouté lors d'un cycle précédent. C'est ce dossier que consulte l'onglet Transformer les données, et le bouton **"Ouvrir le dossier du modèle"** l'ouvre directement dans l'explorateur de fichiers.

### Résumé du corpus (`<nom>_digest.json`)

À chaque entraînement ou amélioration, un résumé condensé de tous les documents utilisés est enregistré à côté du modèle — pas le texte intégral (qui peut être volumineux et redondant), mais pour chaque document : son nom, sa catégorie, un extrait lisible (~500 caractères, coupé proprement sur un mot) et ses mots-clés les plus représentatifs. En TF-IDF, ces mots-clés sont tirés directement des vecteurs déjà calculés pour le regroupement (aucun coût supplémentaire) ; en embeddings, d'une passe TF-IDF légère et indépendante dédiée à l'extraction de mots-clés.

Ce fichier, nettement plus petit que le texte brut, est pensé pour servir de corpus à un futur modèle d'IA local avec un contexte réduit — par exemple pour proposer un nom de catégorie plus pertinent, ou résumer un fichier, sans avoir à ré-extraire ni relire l'intégralité des documents d'origine. Il contient aussi, si l'option est activée (voir Étape 1), la liste des paires de documents quasi identiques détectées (`"duplicates"`).

### Revenir en arrière (historique)

Avant chaque entraînement, amélioration ou renommage, l'état précédent du modèle (`.pkl` + `.json` + `dataset/`) est automatiquement archivé. Dans l'onglet **Transformer les données**, le bouton **"Historique / Revenir en arrière..."** liste ces instantanés et permet d'en restaurer un — l'état courant est lui-même archivé avant, donc une restauration peut toujours être annulée. Le nombre d'instantanés conservés par modèle se règle dans **Paramètres** (`0` désactive l'historique).

## Étape 2 — Transformer les données (optionnel : ajuster les catégories)

Les noms détectés automatiquement (mots-clés TF-IDF, ex. `competences_vitae_techniques`) ne sont pas toujours parlants. Cet onglet se charge automatiquement avec le modèle qui vient d'être entraîné, et présente les catégories dans un tableau (à gauche) avec un panneau de détails (à droite) :

1. **Choisir un modèle** si besoin (même sélecteur que dans les autres onglets — sinon le modèle tout juste entraîné est déjà chargé).
2. Le tableau affiche pour chaque catégorie sa colonne **"Catégorie"** (le nom actuel, éventuellement renommé) et sa colonne **"Nom détecté par le modèle"** (le ou les noms d'origine, basés sur les mots-clés TF-IDF ou KeyBERT, qui ne changent jamais — même après un ou plusieurs renommages, ou un ré-entraînement qui a repris ce nom). Utile pour retrouver ce que le modèle a réellement identifié derrière un nom personnalisé. **Chaque catégorie n'apparaît qu'une seule fois** dans le tableau, même si plusieurs regroupements internes du modèle partagent le même nom affiché (ex. après plusieurs améliorations successives) — leurs noms d'origine respectifs sont alors listés ensemble, séparés par « / ».
3. **Cliquer sur une catégorie** dans le tableau pour voir, à droite, la **liste de ses fichiers** (depuis le dossier `dataset/` du modèle — voir "Où sont rangés les modèles" ci-dessus).
4. **Renommer** : modifier le champ "Nouveau nom" puis cliquer sur "Renommer" — le modèle et son dossier `dataset/` sont mis à jour immédiatement (pas de bouton "Enregistrer" global à part).
5. **"Ouvrir le dossier"** : ouvre le sous-dossier de cette catégorie dans l'explorateur de fichiers.
6. **"Préfixer les fichiers par la catégorie"** : renomme chaque fichier du dataset en le préfixant par le nom de la catégorie (ex. `facture1.pdf` → `Factures_facture1.pdf`). N'affecte que les copies du dataset, jamais les documents d'origine. Sans effet si déjà fait (idempotent).
7. **"Supprimer cette catégorie (→ autre)"** : après confirmation, fusionne la catégorie dans une catégorie fourre-tout **"autre"** — ses documents ne sont jamais perdus, seulement regroupés ailleurs. Indisponible s'il ne reste qu'une seule catégorie, ou sur "autre" elle-même.
8. **"Historique / Revenir en arrière..."** : liste les instantanés du modèle et permet d'y revenir (voir "Revenir en arrière" ci-dessus).

Si le modèle modifié est déjà chargé dans l'onglet Classification, il y est automatiquement rechargé pour refléter les changements.

## Étape 3 — Classification (utiliser le modèle sur de nouveaux documents)

1. **Choisir un modèle** : soit dans la liste **"Modèles disponibles"** (tous les `.pkl` déjà trouvés dans le projet, du plus léger au plus lourd), soit via **"Charger un modèle (.pkl)..."** pour un fichier situé ailleurs.
2. **Ajouter des documents** à classer : glisser-déposer sur la fenêtre, ou boutons "Ajouter des fichiers.../Ajouter un dossier...".
3. Chaque document est analysé et affiché avec sa **catégorie proposée** et un **niveau de confiance** (les prédictions incertaines sont surlignées en jaune, les fichiers illisibles en rouge).
4. **Cliquer sur une ligne** pour voir dans le panneau de droite le **texte que le modèle a effectivement analysé** — utile pour comprendre pourquoi une catégorie a été proposée. Le bouton **"Ouvrir le fichier original"** ouvre le document dans son application habituelle (lecteur PDF, Word...) pour le voir avec sa mise en forme.
5. **Corriger** une catégorie d'un double-clic si besoin.
6. **Choisir le dossier de sortie**, puis **"Valider et classer"** : les fichiers sont rangés dans `<dossier de sortie>/classification_<date>_<heure>/<catégorie>/` (copiés par défaut, ou déplacés si la case correspondante est cochée) — un nouveau sous-dossier horodaté à chaque clic, rien n'est jamais écrasé — et **ce dossier s'ouvre automatiquement** dans l'explorateur de fichiers une fois terminé. Le bouton **"Ouvrir le dernier dossier classé"** permet d'y revenir plus tard. Si deux documents partagent le même nom de fichier, le second reçoit automatiquement un suffixe (`facture_1.pdf`, `facture_2.pdf`...) — aucun fichier n'est jamais silencieusement écrasé par un autre.

**Fichiers "à vérifier" et "non catégorisé"** — deux cases, **"Inclure les fichiers 'à vérifier' dans l'export"** et **"Inclure les fichiers 'non catégorisé' dans l'export"** (cochées par défaut, réglables dans Paramètres), permettent de les exclure de l'export : décochées, ces fichiers restent simplement à leur emplacement d'origine (rien n'est perdu) plutôt que d'être copiés/déplacés.

**Amélioration continue (optionnelle)** — la case **"Améliorer le modèle avec ces documents"** (décochée par défaut, réglable dans Paramètres) fait, après la validation, repasser le modèle chargé par un ré-entraînement. Seuls les documents **réellement corrigés à la main** (double-clic, avec une catégorie différente de la proposition initiale et différente de "a_verifier"/"illisible") y contribuent, comme référence certaine plutôt qu'une simple prédiction : classer un document que le modèle a déjà bien deviné, sans le corriger, ne change rien au modèle. Ce fonctionnement conserve toujours les catégories déjà renommées ou supprimées dans "Transformer les données", et met à jour à chaque fois le dossier `dataset/` du modèle : le lien entre le modèle, ses catégories et leurs fichiers reste donc toujours consultable dans "Transformer les données", même après une amélioration continue déclenchée depuis la Classification. Le modèle est mis à jour en tâche de fond ; une barre de progression s'affiche pendant l'opération.

Une catégorie confirmée à la main reste **acquise pour ce document précis**, même si le regroupement automatique (non supervisé) ne la retrouve pas naturellement : elle apparaît dans "Transformer les données" (avec la mention "confirmée manuellement" si aucun cluster ne porte ce nom), et une classification ultérieure du même document — même dans un nouveau dossier d'export, tant que son contenu est identique — lui redonne systématiquement cette catégorie avec une confiance de 100 %, plutôt que de le laisser retomber sous un nom "naturel" différent au hasard du clustering.

**Documents en double** — deux boutons complètent la liste :
- **"Détecter les doublons"** : compare les documents actuellement chargés (mêmes vecteurs que la prédiction, aucun calcul supplémentaire) et affiche les paires quasi identiques trouvées (similarité ≥ 97 %).
- **"Supprimer les doublons"** (activé après une détection ayant trouvé quelque chose) : déplace les exemplaires en trop vers un dossier **`_backup`** créé à côté du dossier d'origine de chaque fichier — jamais une suppression définitive — et les retire de la liste. Un exemplaire de chaque groupe de quasi-doublons est toujours conservé.

## Étape 4 — Automatisation (classification en continu, sans intervention)

Fait la même chose que la Classification, mais **automatiquement et à intervalle régulier**, sans qu'on ait besoin de rouvrir l'outil à chaque fois.

1. **"Ajouter..."** une automatisation : dossier à surveiller, modèle à appliquer, dossier de sortie, intervalle (**toutes les X minutes / heures / jours**).
2. Elle démarre immédiatement en tâche de fond dès l'ajout.
3. **Plusieurs automatisations peuvent tourner en parallèle** (ex. une pour les factures scannées, une pour les relevés bancaires), chacune avec son propre modèle et son propre dossier.
4. Un fichier déjà traité n'est jamais reclassé deux fois (même en mode "copier"), et la configuration des automatisations est sauvegardée : elles reprennent automatiquement au prochain lancement de l'application.
5. Les cases **"Inclure les fichiers 'à vérifier'"** et **"Inclure les fichiers 'non catégorisé'"** (cochées par défaut, réglables dans Paramètres) permettent de les exclure du dispatch : décochées, ces fichiers restent dans le dossier surveillé et sont **retentés au passage suivant** (utile s'ils finissent par être reconnus avec confiance, par exemple après une amélioration du modèle, ou redeviennent lisibles) plutôt que d'être définitivement ignorés.

Le journal de l'onglet affiche l'historique des passages ("3 nouveau(x) fichier(s) classé(s)", erreurs éventuelles...).

## Étape 5 — Paramètres (configuration technique)

Tous les réglages techniques de l'application sont pilotables depuis cet onglet, et enregistrés dans `config.json` à la racine du projet. Modifier une valeur et cliquer sur **"Enregistrer"** l'applique **immédiatement** aux opérations suivantes, sans redémarrer l'application. **"Réinitialiser aux valeurs par défaut"** remet le formulaire aux valeurs d'origine (à enregistrer pour confirmer).

| Groupe | Réglages |
|---|---|
| Regroupement automatique | nombre minimal/maximal de catégories à essayer, nombre de mots-clés utilisés pour nommer une catégorie, score de silhouette minimal et nombre minimal de documents par catégorie pour accepter un découpage, nommage via KeyBERT activé par défaut, détection des doublons activée par défaut |
| Vectorisation | taille du vocabulaire TF-IDF, taille des n-grammes, modèle d'embeddings par défaut |
| Classification | seuil de confiance minimal, noms des catégories "incertain", "illisible" et "autre" (suppression), cases "Améliorer le modèle" et "Inclure les fichiers à vérifier/non catégorisé dans l'export" cochées par défaut ou non |
| Dossiers de sortie | dossier racine des modèles (`storage/models/`), dossier de sortie par défaut (Classification) |
| Automatisation | fichier de configuration des automatisations, intervalle et mode (copier/déplacer) par défaut, inclusion des fichiers "à vérifier"/"non catégorisé" par défaut |
| Divers | profondeur de recherche des modèles `.pkl`, nombre d'instantanés d'historique conservés par modèle, taille de fenêtre par défaut, port du serveur API local |

**Il n'y a pas de plafond caché au nombre de catégories** : le nombre maximal de catégories détectées par l'entraînement automatique est entièrement piloté par le réglage "nombre maximal de catégories à essayer" — augmentez-le librement si vous avez besoin de plus de catégories. La seule autre limite est mathématique : on ne peut pas former plus de groupes distincts que de documents fournis.

Les réglages bas niveau qui pourraient casser l'extraction ou la vectorisation si mal réglés (expression régulière de tokenisation, liste de mots vides...) restent dans le code plutôt que d'être exposés dans cet onglet.

## Étape 6 — API (piloter l'application depuis un autre programme)

Un serveur HTTP local, entièrement optionnel, expose les mêmes actions que la GUI (entraîner, classer, améliorer un modèle, renommer/supprimer des catégories, consulter et restaurer l'historique) pour qu'un autre programme puisse les appeler directement, sans réimplémenter la logique de l'outil.

1. Dans l'onglet **API**, choisissez un port (par défaut `8756`, modifiable aussi depuis Paramètres) puis cliquez sur **"Démarrer le serveur API"**.
2. Une **clé** est générée à ce moment-là (régénérée à chaque démarrage) et affichée dans l'onglet, avec un bouton **"Copier"**. Elle doit être envoyée dans l'en-tête `Authorization: Bearer <clé>` (ou `X-API-Key: <clé>`) sur toutes les routes sauf `/` et `/health`.
3. La liste complète des routes disponibles (avec leur méthode, leur chemin et le format du corps attendu) s'affiche directement dans l'onglet, et est aussi récupérable via `GET /` sans authentification.
4. Un journal en bas de l'onglet affiche chaque requête reçue en direct.
5. Le serveur ne répond jamais qu'à `127.0.0.1` — il n'est **jamais exposé sur le réseau** — et s'arrête automatiquement à la fermeture de l'application.

Exemple d'appel :

```bash
curl -X POST http://127.0.0.1:8756/classify \
  -H "Authorization: Bearer <clé affichée dans l'onglet API>" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "pdf_fiche_de_paye", "input_dir": "./a_trier", "output_dir": "./classified"}'
```

**N'importe quel programme tournant sur la machine et connaissant la clé peut piloter l'application** : ne la partagez pas, et arrêtez le serveur quand vous n'en avez plus besoin.

---

## Utilisation en ligne de commande

Les trois mêmes étapes existent aussi en CLI, pour l'automatisation via scripts/tâches planifiées :

```bash
# Entraînement (équivalent de l'onglet Entraînement, sans dispatch de fichiers)
python ml_pdf.py discover --input ./mes_documents --output ./classified --model-out model.pkl

# Classification (équivalent de l'onglet Classification)
python ml_pdf.py classify --input ./nouveaux_fichiers --model model.pkl --output ./classified
```

Un mode supplémentaire, réservé à la CLI, permet un entraînement **supervisé** classique quand on préfère définir les catégories soi-même (un sous-dossier = une catégorie, rempli à la main) :

```bash
python ml_pdf.py train --input ./mes_categories --model model.pkl
```

## Compiler un exécutable autonome (sans Python installé sur le poste cible)

`build.ps1` (Windows) et `build.sh` (Linux, y compris depuis WSL) compilent l'application en exécutable autonome avec [PyInstaller](https://pyinstaller.org/) — installez d'abord `pip install -r requirements-build.txt`.

```bash
powershell -File build.ps1        # Windows : dossier ClasseurDocuments/ dans build-windows/dist/
./build.sh                        # Linux : dossier ClasseurDocuments/ dans build-linux/dist/
```

Ajoutez `--onefile` (`--Onefile` pour build.ps1) pour un seul exécutable au lieu d'un dossier — plus simple à distribuer, mais un peu plus lent à démarrer (extraction dans un dossier temporaire à chaque lancement).

Par défaut (variante `lite`), le moteur embeddings (sentence-transformers + torch, ~1 Go) est exclu — c'est déjà le moteur TF-IDF qui est recommandé par défaut dans l'application (voir l'étape 1). Pour l'inclure : `-Variant full` / `--full`.

**Compilation croisée impossible** : un exécutable Windows ne peut être produit que depuis Windows, un exécutable Linux que depuis Linux — chaque script doit tourner sur sa propre plateforme (ou dans WSL pour Linux depuis une machine Windows). Ne jamais lancer les deux scripts en même temps sur un dossier de sortie partagé.

**Nuitka a été essayé puis abandonné** pour ce projet : PyMuPDF embarque un fichier C auto-généré de ~2,3 millions de lignes que MSVC ne peut pas compiler (erreur `C1002`, mémoire interne insuffisante), et MinGW64 casse sur un problème différent d'en-têtes Windows. PyInstaller copie les extensions déjà compilées au lieu de les recompiler depuis la source, évitant structurellement ce type de problème — l'exécutable produit est un peu plus gros et démarre un peu plus lentement, mais le build est nettement plus robuste avec cette pile de dépendances (numpy + scipy + scikit-learn + PyMuPDF).

## Formats de fichiers pris en charge

| Extension | Dépendance |
|---|---|
| `.pdf` `.rtf` | incluse |
| `.txt` `.md` `.csv` `.tsv` `.log` `.json` `.yaml` `.yml` `.ini` `.cfg` `.toml` | incluse |
| `.html` `.htm` `.xml` | incluse (balises retirées automatiquement) |
| `.eml` | incluse |
| `.docx` | `requirements-docx.txt` |
| `.xlsx` `.xlsm` `.pptx` | `requirements-office.txt` |
| `.msg` (email Outlook) | `requirements-msg.txt` |

Non pris en charge : les images (scans sans OCR, `.png`, `.jpg`...) nécessiteraient un moteur OCR externe (ex. Tesseract) en plus de Python — non inclus pour l'instant.

## Comment lire les catégories automatiques

- Un document dont la confiance de prédiction est trop faible (par défaut < 40%) est placé dans **`a_verifier`** plutôt que d'être mal classé en silence.
- Un fichier dont le texte n'a pas pu être extrait (scan sans OCR, PDF chiffré, fichier corrompu...) est placé dans **`non_categorise_texte_illisible`**.
- Si le dossier analysé ne contient en réalité qu'un seul type de document, l'entraînement essaie de ne pas forcer plusieurs catégories artificielles : un découpage qui isolerait des documents presque seuls dans leur coin n'est pas retenu (un cluster à 1 document a toujours une cohésion artificiellement parfaite), et en dessous d'un score de silhouette minimal, une seule catégorie est gardée. Ces deux seuils se règlent dans **Paramètres** (ou au cas par cas dans les paramètres avancés de l'onglet Entraînement). Ce n'est toutefois pas une garantie absolue avec TF-IDF — voir l'encart sur l'échelle du score de silhouette à l'étape 1 : un dossier homogène peut occasionnellement ressortir découpé en 2-3 catégories plutôt qu'une seule. Le cas échéant, fusionnez-les dans l'onglet Transformer les données.

## Structure du projet

```
ml_pdf.py            point d'entrée CLI (discover / train / classify)
ml_pdf_gui.py         point d'entrée de l'interface graphique
pdf_classifier/
  config.py            configuration technique (config.json, onglet Paramètres)
  formats.py            registre des extracteurs de texte par type de fichier
  extraction.py           extraction robuste du texte des documents
  features.py               moteurs de vectorisation (TF-IDF / embeddings)
  discover.py                 détection automatique des catégories (mode "Entraînement")
  train.py                      entraînement supervisé (CLI uniquement)
  classify.py                     application d'un modèle déjà entraîné
  rename.py                         renommage/suppression des catégories (modèle + dataset/)
  automation.py                       jobs planifiés multiples (surveillance de dossier)
  model_store.py                        structure storage/models/<nom>/, sauvegarde atomique, historique/restauration
  utils.py                                écriture atomique, dispatch des fichiers
  gui.py                                    interface graphique (5 onglets)
  cli.py                                      interface en ligne de commande
```
