import webbrowser

def web_search(site_domen):
    url = f"https://{site_domen.lstrip('https://').lstrip('http://')}"
    print(f"Открываю {url}")
    webbrowser.open(url)
    return f"Сайт {site_domen} открыт."