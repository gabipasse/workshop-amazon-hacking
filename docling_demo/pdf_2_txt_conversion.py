from docling.document_converter import DocumentConverter


def converter_pdf_2_txt(path_pdf: str) -> str:
    conversor = DocumentConverter()
    resultado = conversor.convert(path_pdf)
    txt_output = resultado.document.export_to_text()

    return txt_output


def main():
    path_pdf = r"docling_demo\pdf_exemplo_fonte.pdf"
    txt_output = converter_pdf_2_txt(path_pdf)
    print(txt_output)

    path_txt = r"docling_demo\txt_convertido.txt"

    with open(path_txt, "w") as file:
        file.write(txt_output)


if __name__ == "__main__":
    main()
