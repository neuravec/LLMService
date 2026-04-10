"""
examples.py — przykłady użycia llm_service w zwykłym skrypcie Python.

W notebooku wystarczy `await` / `async with`. W .py trzeba użyć
`asyncio.run()` jako entry point. Ten plik pokazuje wszystkie wzorce.

Uruchomienie:
    python examples.py

Wymaga ustawionych zmiennych środowiskowych:
    AZURE_OPENAI_KEY       — klucz API
    AZURE_OPENAI_ENDPOINT  — endpoint (np. https://my-resource.openai.azure.com)
    AZURE_OPENAI_MODEL     — deployment name (np. gpt-4.1)

Albo pliku config.yaml (patrz config.example.yaml).
"""

import asyncio
import os
import sys

from pydantic import BaseModel, Field

from llm_service import (
    LLMClient,
    LLMConfig,
    LLMError,
    Pipeline,
    Step,
    detect_capabilities,
)


# ---------------------------------------------------------------------------
# Konfiguracja — wybierz sposób A lub B
# ---------------------------------------------------------------------------

def get_config() -> LLMConfig:
    """Załaduj config z YAML lub ze zmiennych środowiskowych."""

    # Sposób A: z pliku YAML
    if os.path.exists("config.yaml"):
        print("[config] Ładuję z config.yaml")
        return LLMConfig.from_yaml("config.yaml")

    # Sposób B: ze zmiennych środowiskowych
    print("[config] Ładuję ze zmiennych środowiskowych")
    return LLMConfig(
        api_key=os.environ["AZURE_OPENAI_KEY"],
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        model_name=os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1"),
    )


# ===================================================================
# Przykład 1: Proste zapytanie
# ===================================================================

async def example_chat(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 1: Proste zapytanie (chat)")
    print("=" * 60)

    async with LLMClient(cfg) as llm:
        answer = await llm.chat(
            "Czym jest Azure OpenAI Service? Odpowiedz w 2 zdaniach.",
            system="Odpowiadaj po polsku, zwięźle.",
        )
        print(answer)


# ===================================================================
# Przykład 2: Batch — wiele promptów równolegle
# ===================================================================

async def example_batch(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 2: Batch (5 promptów równolegle)")
    print("=" * 60)

    prompts = [
        "Czym jest ETL?",
        "Czym jest RAG w kontekście LLM?",
        "Co to jest vector database?",
        "Wyjaśnij pojęcie fine-tuning.",
        "Czym jest prompt engineering?",
    ]

    async with LLMClient(cfg) as llm:
        results = await llm.batch(
            prompts,
            system="Odpowiadaj w 1-2 zdaniach, po polsku.",
        )

    for q, a in zip(prompts, results):
        print(f"\nQ: {q}\nA: {a}")


# ===================================================================
# Przykład 3: JSON output
# ===================================================================

async def example_json(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 3: JSON output (bez schematu)")
    print("=" * 60)

    async with LLMClient(cfg) as llm:
        data = await llm.chat_json(
            "Podaj 3 największe miasta w Polsce z populacją. "
            "Zwróć JSON z kluczem 'cities', każde miasto to obiekt z 'name' i 'population'.",
        )
        print(data)


# ===================================================================
# Przykład 4: Structured output z Pydantic
# ===================================================================

class City(BaseModel):
    name: str = Field(description="Nazwa miasta")
    country: str = Field(description="Kraj")
    population: int = Field(description="Populacja")


class CityList(BaseModel):
    cities: list[City] = Field(description="Lista miast")


async def example_structured(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 4: Structured output (Pydantic)")
    print("=" * 60)

    async with LLMClient(cfg) as llm:
        result = await llm.structured(
            "Podaj 5 największych miast w Europie.",
            CityList,
            system="Zwracaj dokładne dane.",
        )

    for city in result.cities:
        print(f"  {city.name} ({city.country}): {city.population:,}")


# ===================================================================
# Przykład 5: Ekstrakcja danych z dokumentu
# ===================================================================

class InvoiceData(BaseModel):
    invoice_number: str = Field(description="Numer faktury")
    seller: str = Field(description="Nazwa sprzedawcy")
    buyer: str = Field(description="Nazwa kupującego")
    total_amount: float = Field(description="Kwota brutto")
    currency: str = Field(description="Waluta")
    issue_date: str = Field(description="Data wystawienia (YYYY-MM-DD)")
    items_count: int = Field(description="Liczba pozycji na fakturze")


async def example_extraction(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 5: Ekstrakcja danych z faktury")
    print("=" * 60)

    document = """
    FAKTURA VAT nr FV/2026/04/0137
    Data wystawienia: 2026-04-08

    Sprzedawca: TechSoft Sp. z o.o., ul. Marszałkowska 10, Warszawa
    Nabywca: DataCorp S.A., ul. Krakowska 55, Kraków

    Pozycje:
    1. Licencja Enterprise - 12 000,00 PLN netto
    2. Wdrożenie systemu  -  8 500,00 PLN netto
    3. Szkolenie zespołu  -  3 200,00 PLN netto

    Razem netto: 23 700,00 PLN
    VAT 23%:      5 451,00 PLN
    Razem brutto: 29 151,00 PLN
    """

    async with LLMClient(cfg) as llm:
        invoice = await llm.structured(
            f"Wyciągnij dane z tej faktury:\n\n{document}",
            InvoiceData,
            system="Ekstrahujesz ustrukturyzowane dane z dokumentów.",
        )

    print(f"  Faktura:    {invoice.invoice_number}")
    print(f"  Sprzedawca: {invoice.seller}")
    print(f"  Kupujący:   {invoice.buyer}")
    print(f"  Kwota:      {invoice.total_amount} {invoice.currency}")
    print(f"  Data:       {invoice.issue_date}")
    print(f"  Pozycje:    {invoice.items_count}")


# ===================================================================
# Przykład 6: Batch structured — wiele dokumentów naraz
# ===================================================================

class ContractData(BaseModel):
    contract_number: str = Field(description="Numer umowy")
    date: str = Field(description="Data zawarcia (YYYY-MM-DD)")
    party_a: str = Field(description="Pierwsza strona umowy")
    party_b: str = Field(description="Druga strona umowy")
    amount: float = Field(description="Kwota netto")
    currency: str = Field(description="Waluta")


async def example_batch_structured(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 6: Batch structured (3 umowy równolegle)")
    print("=" * 60)

    documents = [
        "Umowa nr U/2026/001 zawarta dnia 2026-01-15 pomiędzy ABC Sp. z o.o. a XYZ S.A. na kwotę 150 000 PLN netto.",
        "Umowa nr U/2026/002 zawarta dnia 2026-02-20 pomiędzy DEF Sp. z o.o. a GHI S.A. na kwotę 85 000 EUR netto.",
        "Umowa nr U/2026/003 zawarta dnia 2026-03-10 pomiędzy JKL Sp. z o.o. a MNO S.A. na kwotę 220 000 PLN netto.",
    ]

    prompts = [f"Wyciągnij dane z tej umowy:\n\n{doc}" for doc in documents]

    async with LLMClient(cfg) as llm:
        contracts = await llm.batch_structured(
            prompts,
            ContractData,
            system="Ekstrahujesz dane z umów.",
        )

    for c in contracts:
        print(f"  {c.contract_number}: {c.party_a} <-> {c.party_b} -- {c.amount:,.0f} {c.currency}")


# ===================================================================
# Przykład 7: Obsługa błędów (LLMError)
# ===================================================================

async def example_error_handling(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 7: Obsługa błędów (LLMError)")
    print("=" * 60)

    # Celowo zły config żeby pokazać error handling
    bad_cfg = LLMConfig(
        api_key="bad-key",
        endpoint=cfg.endpoint,
        model_name=cfg.model_name,
        retries=1,
    )

    try:
        async with LLMClient(bad_cfg) as llm:
            await llm.chat("test")
    except LLMError as e:
        print(f"Złapano LLMError:")
        print(e)
        print(f"\n  Bezpośredni dostęp do pól:")
        print(f"    status_code: {e.status_code}")
        print(f"    error_code:  {e.error_code}")
        print(f"    hint:        {e.hint}")
    except Exception as e:
        # Fallback na wypadek problemów z połączeniem
        print(f"Inny błąd: {type(e).__name__}: {e}")


# ===================================================================
# Przykład 8: Auto-detekcja modeli (reasoning vs standard)
# ===================================================================

def example_model_detection() -> None:
    print("\n" + "=" * 60)
    print("Przykład 8: Auto-detekcja modeli")
    print("=" * 60)

    models = ["gpt-4.1", "gpt-4.1-mini", "gpt-5.4-mini", "o3", "o4-mini"]

    for name in models:
        caps = detect_capabilities(name)
        mode = "REASONING" if caps.reasoning else "STANDARD"
        print(f"  {name:20s} -> {mode:10s} temp={caps.supports_temperature}  system={caps.supports_system_message}")


# ===================================================================
# Przykład 9: Pipeline — fan-out + fan-in
# ===================================================================

class BusinessAnalysis(BaseModel):
    strengths: list[str] = Field(description="Mocne strony")
    risks: list[str] = Field(description="Ryzyka")

class LegalAnalysis(BaseModel):
    compliance_issues: list[str] = Field(description="Problemy z compliance")
    recommendations: list[str] = Field(description="Rekomendacje prawne")

class CriticVerdict(BaseModel):
    contradictions: list[str] = Field(description="Sprzeczności między analizami")
    final_recommendation: str = Field(description="Końcowa rekomendacja")
    confidence: str = Field(description="Poziom pewności: low/medium/high")


async def example_pipeline(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 9: Pipeline (biznes + prawnik -> krytyk)")
    print("=" * 60)

    pipe = Pipeline(
        # Warstwa 1: równolegle
        Step(
            name="biznes",
            system="Jesteś analitykiem biznesowym. Oceń mocne strony i ryzyka.",
            output=BusinessAnalysis,
        ),
        Step(
            name="prawnik",
            system="Jesteś prawnikiem korporacyjnym. Oceń ryzyka prawne.",
            output=LegalAnalysis,
        ),
        # Warstwa 2: czeka na obu
        Step(
            name="krytyk",
            system="Jesteś krytycznym recenzentem. Porównaj analizy.",
            prompt=(
                "Oto analizy do oceny:\n\n"
                "--- ANALIZA BIZNESOWA ---\n{biznes}\n\n"
                "--- ANALIZA PRAWNA ---\n{prawnik}\n\n"
                "Oryginalny dokument:\n{input}"
            ),
            output=CriticVerdict,
            depends_on=["biznes", "prawnik"],
        ),
    )

    proposal = (
        "Firma XYZ proponuje przejęcie 60% udziałów w startup ABC za 5 mln PLN. "
        "ABC ma roczne przychody 1.2 mln PLN ale jest nierentowna (strata 300k PLN rocznie). "
        "ABC posiada 3 patenty na technologię AI do analizy dokumentów medycznych. "
        "Due diligence ujawnił niezapłacone zobowiązania podatkowe za 2024 rok (180k PLN)."
    )

    async with LLMClient(cfg) as llm:
        result = await pipe.run(llm, input=proposal)

    biz = result.output("biznes")
    law = result.output("prawnik")
    critic = result.output("krytyk")

    print(f"\n  Mocne strony:  {biz.strengths}")
    print(f"  Ryzyka biznes: {biz.risks}")
    print(f"  Compliance:    {law.compliance_issues}")
    print(f"  Rekomendacje:  {law.recommendations}")
    print(f"  Sprzeczności:  {critic.contradictions}")
    print(f"  Werdykt:       {critic.final_recommendation}")
    print(f"  Pewność:       {critic.confidence}")
    print(f"\n{result.trace()}")


# ===================================================================
# Przykład 10: Pipeline na wielu dokumentach (batch + pipeline)
# ===================================================================

class DocSummary(BaseModel):
    title: str = Field(description="Tytuł/temat dokumentu")
    key_facts: list[str] = Field(description="Kluczowe fakty")
    sentiment: str = Field(description="positive/neutral/negative")

class QACheck(BaseModel):
    is_accurate: bool = Field(description="Czy podsumowanie jest dokładne")
    missing_info: list[str] = Field(description="Brakujące informacje")


async def example_batch_pipeline(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 10: Pipeline na wielu dokumentach równolegle")
    print("=" * 60)

    qa_pipe = Pipeline(
        Step(
            name="summary",
            system="Podsumowujesz dokumenty. Wyciągaj kluczowe fakty.",
            output=DocSummary,
        ),
        Step(
            name="qa",
            system="Sprawdzasz jakość podsumowań. Porównujesz z oryginałem.",
            prompt="Podsumowanie:\n{summary}\n\nOryginalny dokument:\n{input}\n\nCzy podsumowanie jest kompletne?",
            output=QACheck,
            depends_on=["summary"],
        ),
    )

    documents = [
        "Raport Q1 2026: przychody wzrosły o 15% r/r do 45 mln PLN. EBITDA: 12 mln PLN.",
        "Notatka ze spotkania: zespół zdecydował o migracji do Kubernetes. Deadline: koniec Q2.",
        "Reklamacja klienta: system nie działa od 3 dni. Klient grozi wypowiedzeniem umowy.",
    ]

    async with LLMClient(cfg) as llm:
        all_results = await asyncio.gather(*[
            qa_pipe.run(llm, input=doc) for doc in documents
        ])

    for i, res in enumerate(all_results):
        summary = res.output("summary")
        qa = res.output("qa")
        print(f"\n  --- Dokument {i + 1} ---")
        print(f"  Tytuł:    {summary.title}")
        print(f"  Sentiment:{summary.sentiment}")
        print(f"  QA OK:    {qa.is_accurate}")
        if qa.missing_info:
            print(f"  Braki:    {qa.missing_info}")


# ===================================================================
# Przykład 11: Śledzenie tokenów i kosztów
# ===================================================================

async def example_usage_tracking(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 11: Śledzenie tokenów i kosztów")
    print("=" * 60)

    async with LLMClient(cfg) as llm:
        await llm.chat("Czym jest Python?")
        await llm.chat("Czym jest Rust?")
        await llm.batch(["Czym jest Go?", "Czym jest Java?"])

        print(llm.usage.summary())
        print()
        print(f"  Requesty:       {llm.usage.request_count}")
        print(f"  Prompt tokens:  {llm.usage.prompt_tokens:,}")
        print(f"  Compl. tokens:  {llm.usage.completion_tokens:,}")
        print(f"  Total tokens:   {llm.usage.total_tokens:,}")
        print(f"  Koszt:          ${llm.usage.cost_usd:.4f} USD")


# ===================================================================
# Przykład 12: Cennik modeli
# ===================================================================

def example_pricing() -> None:
    print("\n" + "=" * 60)
    print("Przykład 12: Cennik modeli")
    print("=" * 60)

    from llm_service import get_pricing

    models = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "o3", "o4-mini"]
    print(f"  {'Model':<20} {'Input $/1M':>12} {'Output $/1M':>12}")
    print(f"  {'-' * 46}")
    for name in models:
        p = get_pricing(name)
        if p:
            print(f"  {name:<20} ${p[0]:>10.2f} ${p[1]:>10.2f}")


# ===================================================================
# Przykład 13: Vision — opis obrazu
# ===================================================================

async def example_vision(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 13: Vision — opis obrazu")
    print("=" * 60)

    # Wymaga pliku z obrazem — zakomentowane by nie crashowało
    # async with LLMClient(cfg) as llm:
    #     answer = await llm.chat(
    #         "Opisz co widzisz na tym obrazie.",
    #         images=["path/to/image.png"],
    #     )
    #     print(answer)
    #     print(f"\n{llm.usage.summary()}")

    # Demo: jak wygląda content z obrazem (bez API call)
    from llm_service import build_content_parts
    content = build_content_parts(
        "Opisz ten obraz",
        images=["https://example.com/photo.jpg"],
    )
    print(f"  Content type: {type(content).__name__}")
    print(f"  Parts: {len(content)}")
    print(f"  Text part: {content[0]}")
    print(f"  Image part type: {content[1]['type']}")
    print(f"  Image URL: {content[1]['image_url']['url'][:50]}...")


# ===================================================================
# Przykład 14: Vision + Structured — OCR ze skanu
# ===================================================================

class ScannedInvoice(BaseModel):
    invoice_number: str = Field(description="Numer faktury")
    seller: str = Field(description="Nazwa sprzedawcy")
    buyer: str = Field(description="Nazwa kupującego")
    total_gross: float = Field(description="Kwota brutto")
    currency: str = Field(description="Waluta")


async def example_vision_structured(cfg: LLMConfig) -> None:
    print("\n" + "=" * 60)
    print("Przykład 14: Vision + Structured — OCR ze skanu")
    print("=" * 60)

    # Wymaga pliku — pokaz tylko budowy requestu
    # async with LLMClient(cfg) as llm:
    #     invoice = await llm.structured(
    #         "Wyciągnij dane z tej zeskanowanej faktury.",
    #         ScannedInvoice,
    #         images=["scan.jpg"],
    #         image_detail="high",
    #     )
    #     print(f"Faktura: {invoice.invoice_number}")

    print("  (wymaga pliku z obrazem — odkomentuj w kodzie)")
    print()
    print("  Użycie:")
    print('    invoice = await llm.structured(')
    print('        "Wyciągnij dane z tej faktury.",')
    print('        ScannedInvoice,')
    print('        images=["scan.jpg"],')
    print('        image_detail="high",')
    print('    )')


# ===================================================================
# Main — uruchom wybrane przykłady
# ===================================================================

async def main() -> None:
    cfg = get_config()

    # Nie wymagają API
    example_model_detection()
    example_pricing()

    # Wymagają połączenia z Azure OpenAI
    await example_chat(cfg)
    await example_batch(cfg)
    await example_json(cfg)
    await example_structured(cfg)
    await example_extraction(cfg)
    await example_batch_structured(cfg)
    await example_error_handling(cfg)
    await example_pipeline(cfg)
    await example_batch_pipeline(cfg)
    await example_usage_tracking(cfg)

    # Vision — demo bez API
    await example_vision(cfg)
    await example_vision_structured(cfg)


if __name__ == "__main__":
    # W .py używamy asyncio.run() jako entry point
    asyncio.run(main())
