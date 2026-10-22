"""Captura una app Gradio local con Playwright.

Abre la URL, opcionalmente elige un valor del Dropdown, escribe cada pregunta,
espera a que Gradio termine de responder y guarda un PNG de la página completa.

Uso:
    python scripts/capture_gradio.py http://127.0.0.1:7861 evidencias/A1_gradio.png \
        --question "¿Cuál es el precio del plan Enterprise?" --question "..."
"""
import argparse

from playwright.sync_api import sync_playwright


def wait_until_idle(page, timeout_ms: int) -> None:
    # Gradio limpia el textbox al terminar chat() y quita las clases de carga.
    page.wait_for_function(
        """() => {
            const box = document.querySelector('textarea');
            const busy = document.querySelectorAll('.generating, .pending, .eta-bar');
            return box && box.value === '' && busy.length === 0;
        }""",
        timeout=timeout_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("out")
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--dropdown", default=None, help="Texto de la opción a elegir")
    parser.add_argument("--dropdown-label", default="Paper", help="Etiqueta del Dropdown")
    parser.add_argument("--timeout", type=int, default=180, help="segundos por pregunta")
    args = parser.parse_args()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_selector("textarea")

        if args.dropdown:
            page.get_by_label(args.dropdown_label).first.click()
            page.get_by_role("option", name=args.dropdown, exact=True).click()

        for question in args.question:
            page.locator("textarea").first.fill(question)
            page.get_by_role("button", name="Enviar").click()
            page.wait_for_timeout(1500)
            wait_until_idle(page, args.timeout * 1000)
            page.wait_for_timeout(1000)

        page.screenshot(path=args.out, full_page=True)
        browser.close()
    print(f"captura guardada en {args.out}")


if __name__ == "__main__":
    main()
