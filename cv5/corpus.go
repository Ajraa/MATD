package main

// Document reprezentuje jeden dokument v korpusu.
type Document struct {
	ID      int
	Title   string
	Content string
}

// Korpus 52 českých dokumentů z oblasti informatiky.
var corpus = []Document{
	{1, "Počítačové sítě", "Internet je globální síť propojující miliony počítačů na celém světě. Protokol TCP/IP tvoří základ přenosu dat přes internet. Směrovače přeposílají pakety mezi různými sítěmi."},
	{2, "Operační systémy", "Linux je open-source operační systém používaný na serverech i osobních počítačích. Jádro systému spravuje paměť, procesy a vstupně-výstupní operace. Windows dominuje trhu osobních počítačů."},
	{3, "Databázové systémy", "Relační databáze ukládají data ve formě tabulek s řádky a sloupci. SQL je standardní jazyk pro dotazování relačních databází. Indexy urychlují vyhledávání v databázi."},
	{4, "Umělá inteligence", "Strojové učení umožňuje počítačům učit se z dat bez explicitního programování. Neuronové sítě jsou inspirovány biologickými neurony lidského mozku. Hluboké učení dosáhlo průlomu v rozpoznávání obrazu."},
	{5, "Programovací jazyky", "Python je populární jazyk vhodný pro datovou vědu a strojové učení. Go byl vyvinut společností Google a klade důraz na jednoduchost a výkon. Java běží na virtuálním stroji JVM a je platformově nezávislá."},
	{6, "Kybernetická bezpečnost", "Šifrování chrání data před neoprávněným přístupem pomocí kryptografických algoritmů. Firewall filtruje síťový provoz a blokuje nebezpečná spojení. Phishing je technika sociálního inženýrství k získání hesel."},
	{7, "Cloudové výpočty", "Cloud computing umožňuje pronájem výpočetních zdrojů přes internet. Amazon Web Services, Microsoft Azure a Google Cloud jsou přední poskytovatelé cloudu. Kontejnery Docker zjednodušují nasazení aplikací."},
	{8, "Vývoj softwaru", "Agilní metodiky jako Scrum umožňují iterativní vývoj softwaru. Git je systém pro správu verzí zdrojového kódu. Průběžná integrace automatizuje testování a nasazení kódu."},
	{9, "Počítačová grafika", "Grafické procesory GPU jsou specializované obvody pro renderování obrazu. OpenGL a DirectX jsou rozhraní pro programování grafiky. Ray tracing simuluje fyzikální chování světla pro realistický obraz."},
	{10, "Algoritmy a datové struktury", "Halda je stromová datová struktura vhodná pro prioritní frontu. Quicksort je efektivní řadící algoritmus s průměrnou složitostí O(n log n). Hashovací tabulka umožňuje vyhledávání v konstantním čase."},
	{11, "Robotika", "Roboti jsou stroje schopné vykonávat úkoly autonomně nebo pod lidskou kontrolou. Průmysloví roboti nahrazují manuální práci na výrobních linkách. Autonomní vozidla využívají senzory lidar a kamery pro navigaci."},
	{12, "Internetová architektura", "DNS překládá doménová jména na IP adresy počítačů. HTTP a HTTPS jsou protokoly pro přenos webových stránek. Webové servery jako Apache nebo Nginx obsluhují požadavky klientů."},
	{13, "Mobilní technologie", "Smartphony kombinují funkce telefonu, fotoaparátu a počítače. Android je nejrozšířenější mobilní operační systém na světě. Apple iOS je uzavřený systém pro zařízení iPhone a iPad."},
	{14, "Datová věda", "Analýza dat odhaluje skryté vzory ve velkých souborech informací. Pandas a NumPy jsou základní knihovny pro práci s daty v Pythonu. Vizualizace dat pomáhá komunikovat výsledky analýzy."},
	{15, "Kvantové počítání", "Kvantové počítače využívají principy kvantové mechaniky pro výpočty. Qubit může existovat v superpozici stavů 0 a 1 současně. Kvantové algoritmy mohou prolomit současné šifrovací metody."},
	{16, "Blockchain technologie", "Blockchain je distribuovaná databáze tvořená řetězcem bloků. Bitcoin je první a nejznámější kryptoměna postavená na blockchainu. Ethereum umožňuje spouštění chytrých kontraktů na blockchainu."},
	{17, "Internet věcí", "IoT propojuje každodenní předměty s internetem a umožňuje jejich vzdálené ovládání. Chytrá domácnost využívá senzory a automatizaci pro úsporu energie. Průmyslový IoT zvyšuje efektivitu výrobních procesů."},
	{18, "Zpracování přirozeného jazyka", "NLP umožňuje počítačům porozumět lidskému jazyku a pracovat s textem. Modely GPT generují koherentní text na základě trénovacích dat. Strojový překlad automaticky převádí text mezi jazyky."},
	{19, "Počítačové vidění", "Rozpoznávání obrazu identifikuje objekty na fotografiích a videích. Konvoluční neuronové sítě jsou standardem pro analýzu obrazu. Detekce obličeje se používá v bezpečnostních systémech i smartphonech."},
	{20, "Herní průmysl", "Videohry jsou interaktivní zábavní software pro počítače a konzole. Unreal Engine a Unity jsou populární herní enginy pro vývoj her. Virtuální realita vytváří imerzivní zážitek prostřednictvím VR brýlí."},
	{21, "Sítě a protokoly", "IPv6 rozšiřuje adresní prostor internetu na biliony adres. Wi-Fi umožňuje bezdrátové připojení k síti v dosahu přístupového bodu. Bluetooth slouží ke krátkodosahové bezdrátové komunikaci mezi zařízeními."},
	{22, "Programování webu", "HTML definuje strukturu webové stránky pomocí značek. CSS stylizuje vzhled webových stránek a určuje jejich vizuální prezentaci. JavaScript přidává interaktivitu a dynamické chování webových stránek."},
	{23, "Mikroprocesorová technika", "CPU je mozek počítače provádějící instrukce programů. Moderní procesory obsahují více jader pro paralelní zpracování úloh. Cache paměť urychluje přístup k často používaným datům."},
	{24, "Softwarové testování", "Jednotkové testy ověřují správnost jednotlivých funkcí nebo metod kódu. Integrační testování ověřuje spolupráci různých komponent systému. Automatizované testy zvyšují spolehlivost softwaru a urychlují vývoj."},
	{25, "Strojové učení", "Regrese předpovídá numerické hodnoty na základě vstupních příznaků. Klasifikace přiřazuje vstupní data do předem definovaných kategorií. Shluková analýza hledá skupiny podobných datových bodů bez učitele."},
	{26, "Paralelní výpočty", "Vícevláknové programování umožňuje paralelní provádění více úloh najednou. MapReduce je programovací model pro zpracování velkých datových sad. GPU jsou vhodné pro masivně paralelní výpočty v strojovém učení."},
	{27, "Operační paměť", "RAM je rychlá volatilní paměť pro dočasné uložení spuštěných programů. Pevné disky SSD jsou rychlejší alternativou k tradičním magnetickým diskům. Hierarchie paměti vyvažuje rychlost a kapacitu úložišť."},
	{28, "Síťová bezpečnost", "VPN šifruje síťový provoz a skrývá IP adresu uživatele. TLS protokol zajišťuje bezpečný přenos dat přes internet. Certifikáty SSL ověřují identitu webových serverů a šifrují komunikaci."},
	{29, "Softwarová architektura", "Mikroslužby rozdělují aplikaci na malé nezávislé služby s vlastní zodpovědností. REST API umožňuje komunikaci mezi klientem a serverem přes HTTP. Návrhové vzory nabízejí osvědčená řešení pro opakující se problémy."},
	{30, "Kompilátory a interprety", "Kompilátor překládá zdrojový kód do strojového kódu před spuštěním. Interpret vykonává zdrojový kód řádek po řádku za běhu programu. LLVM je infrastruktura kompilátorů podporující mnoho programovacích jazyků."},
	{31, "Bioinformatika", "Bioinformatika kombinuje biologii, počítačové vědy a statistiku. Sekvenování genomu odhaluje pořadí nukleotidů v DNA organismů. Algoritmy pro zarovnání sekvencí hledají podobnosti mezi genetickými řetězci."},
	{32, "Teorie grafů", "Graf je matematická struktura tvořená vrcholy a hranami. Dijkstrův algoritmus hledá nejkratší cestu v ohodnoceném grafu. Stromy jsou speciální případ grafů bez cyklů."},
	{33, "Funkcionální programování", "Funkcionální jazyky jako Haskell nebo Erlang kladou důraz na neměnnost dat. Vyšší funkce přijímají nebo vrací jiné funkce jako argumenty. Rekurze nahrazuje cykly ve funkcionálním programování."},
	{34, "Embedded systémy", "Vestavěné systémy jsou specializované počítače integrované do zařízení. Arduino je populární platforma pro prototypování embedded projektů. Real-time operační systémy zajišťují deterministické časování operací."},
	{35, "Informační vyhledávání", "Invertovaný index mapuje termíny na dokumenty obsahující dané slovo. TF-IDF měří důležitost slova v dokumentu vzhledem k celému korpusu. Vektorizace dokumentů umožňuje výpočet podobnosti textů."},
	{36, "Zpracování signálů", "Fourierova transformace rozkládá signál na frekvenční složky. Filtry odstraňují šum a nežádoucí frekvence ze signálu. Digitální zpracování signálů se využívá v audio, videu i komunikacích."},
	{37, "Formální jazyky", "Automaty modelují výpočetní procesy s konečným počtem stavů. Regulární výrazy popisují vzory v textových řetězcích. Chomského hierarchie klasifikuje gramatiky podle jejich vyjadřovací síly."},
	{38, "Numerické metody", "Metoda nejmenších čtverců minimalizuje součet čtverců residuí. Metoda Newton-Raphson iterativně hledá kořeny nelineárních rovnic. Numerická integrace aproximuje hodnotu určitého integrálu."},
	{39, "Distribuované systémy", "Konsensus algoritmy zajišťují shodu uzlů v distribuovaném systému. CAP teorém říká, že distribuovaný systém nemůže mít současně konzistenci, dostupnost a odolnost proti rozdělení. Apache Kafka je platforma pro streamování událostí."},
	{40, "Kryptografie", "Asymetrická kryptografie používá pár klíčů: veřejný a soukromý. RSA algoritmus je základem mnoha šifrovacích protokolů. Hash funkce SHA-256 generuje otisk zprávy fixní délky."},
	{41, "Analýza dat", "Pivot tabulky agregují a sumarizují velká množství dat. SQL dotazy filtrují a spojují data z relačních databází. Strojové učení automatizuje hledání vzorů v datových sadách."},
	{42, "Webové rámce", "Django je komplexní Python framework pro vývoj webových aplikací. React je JavaScriptová knihovna pro tvorbu interaktivních uživatelských rozhraní. Spring Boot zjednodušuje vývoj Java webových aplikací."},
	{43, "Síťové protokoly", "HTTP/2 přináší multiplexování požadavků a komprimaci hlaviček. gRPC je moderní RPC rámec pro komunikaci mikroslužeb. WebSocket umožňuje obousměrnou komunikaci v reálném čase."},
	{44, "Lineární algebra", "Matice jsou základní nástroj pro reprezentaci lineárních transformací. Vlastní vektory matice se transformací nemění, pouze škálují. SVD rozklad matice se využívá v doporučovacích systémech."},
	{45, "Zpracování textu", "Tokenizace rozděluje text na základní jednotky zvané tokeny. Lemmatizace převádí slova na jejich základní tvar. TF-IDF přiřazuje váhy slovům podle jejich důležitosti v dokumentu."},
	{46, "Výpočetní složitost", "Třídy P a NP popisují problémy řešitelné v polynomiálním čase. NP-úplné problémy jsou nejtěžší problémy ve třídě NP. Algoritmy s exponenciální složitostí jsou pro velké vstupy nepoužitelné."},
	{47, "Strojový překlad", "Neuronové sítě Transformer revolutionizovaly strojový překlad textu. Pozornostní mechanismus umožňuje modelu zaměřit se na relevantní části vstupu. BLEU skóre měří kvalitu strojového překladu porovnáním s referenčními překlady."},
	{48, "Logické programování", "Prolog je deklarativní jazyk pro logické programování. Fakta a pravidla definují znalostní bázi v Prologu. Unifikace je základní operace pro porovnání termů v logickém programování."},
	{49, "Počítačová architektura", "Von Neumannova architektura odděluje paměť programu od paměti dat. Pipeline zpracovává více instrukcí současně v různých fázích. RISC procesory mají jednoduchý instrukční soubor pro rychlé vykonání."},
	{50, "Velká data", "Hadoop je framework pro distribuované zpracování velkých datových sad. Spark umožňuje rychlé in-memory zpracování dat v clusteru. NoSQL databáze jako MongoDB jsou vhodné pro nestrukturovaná data."},
	{51, "Strojové vidění", "YOLO algoritmus detekuje objekty v obraze v reálném čase. Segmentace obrazu přiřazuje každému pixelu třídu nebo identitu objektu. Transfer learning přenáší znalosti z natrénovaného modelu na nový úkol."},
	{52, "Telekomunikace", "5G síť nabízí nízkou latenci a vysokou přenosovou rychlost. Optická vlákna přenášejí data rychlostí světla na velké vzdálenosti. Satelitní internet poskytuje připojení i ve vzdálených oblastech světa."},
}
