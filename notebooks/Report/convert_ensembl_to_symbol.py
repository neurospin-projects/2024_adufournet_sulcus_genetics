import requests

def get_gene_symbol(ensembl_id):
    
    url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print(data)
        return data.get("display_name", 'Symbol not found')
    # .get(key, default_value = None)
    else:
        return "Error: Not found"

        
ensembl_id = 'ENSG00000146535'
gene_symbol = get_gene_symbol(ensembl_id)
print(f'The gene symbol for {ensembl_id} is: {gene_symbol}')