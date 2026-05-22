from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Table, TableStyle
from reportlab.platypus import Image as ReportLabImage
from collections import Counter
import os
import pandas as pd
import requests


def get_gene_symbol(ensembl_id):
    
    url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("display_name", 'Symbol not found')
    else:
        return "Symbol not found"

path_to_Champollion = "/home/ad279118/ukb/26irene_AD_UKB_optim/results/Champollion_V1_32"
Champollion_version = "32PCs_TIV_20genpcs"
nb_dim=  32
PCA = "with PCA" #reducting the latent space to 32 dimensions
folder = "32PCs" #32PCs
population="White"
path_to_model="/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation"
# "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation"
# "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation_latent_256"
nb_geneticPC=20

# Setup for the report
with open(f"{path_to_Champollion}/list_model_32PCs.txt") as f:
    regions_models = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

regions_models.sort()
#regions_models = regions_models[:3]
print(regions_models)

class MyDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        # Automatically bookmark Paragraphs with style 'Title'
        if isinstance(flowable, Paragraph) and flowable.style.name == 'Title':
            key = flowable.getPlainText()
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(key, key, level=0, closed=False)

# Output PDF
output_file = f"{path_to_Champollion}/Champollion_V1_{Champollion_version}_summary.pdf"

# Use your custom doc template
document = MyDocTemplate(output_file, pagesize=A4)

# Get A4 page dimensions
page_width, page_height = A4
max_width = page_width - 100  # margins
max_height = page_height - 100
elements = []

# Define a style for the title
style_title = ParagraphStyle(
    "Title",
    fontName="Helvetica",
    fontSize=16,
    spaceAfter=12,
    textColor=colors.black
)

style_subtitle = ParagraphStyle(
    "Title",
    fontName="Helvetica",
    fontSize=12,
    spaceAfter=10,
    textColor=colors.black
)

# Define a style for the content
style_normal = getSampleStyleSheet()["Normal"]

# Main report title
main_title = f"Champollion V1 (initialy {nb_dim} dimensions) results, {PCA}"
elements.append(Paragraph(main_title, ParagraphStyle(
    name="MainTitle",
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    spaceAfter=20,
    textColor=colors.black,
    alignment=1  # centered
)))

# Methodology description
methodology_text = """
<b>Les modèles utilisés proviennent de:</b><br/>
{path_to_model}<br/>
(Entraînement sur 42,433 sujets).<br/><br/>

<b>Chacune des dimensions est résidualisée en utilisant la formule :</b><br/>
<font name="Courier">{dim_formula}</font><br/><br/>

<b>plink2-17-02-22 :</b><br/>
--snps-only<br/>
--maf 0.01<br/>
--max-alleles 2<br/>
--keep {{selected IID based on white.British.ancestry}}<br/>
--geno<br/>
--mind<br/>
--hwe 1e-9<br/><br/>

Dans le cas de <b>white.British.ancestry</b>, 35,940 sujets sont détectés dans le bfile.<br/>
<b>Seuls les résultats issus de la cohorte de découverte</b> (white.British.ancestry) sont présents dans le PDF.<br/><br/>

Le nom de la région d'intérêt, suivie du modèle qui a été utilisé pour la représentation associée (par exemple: CINGULATE_left — name17-24-32_191)<br/><br/>

1. Tableau des résultats résumés de l'analyse LDSC (v2.0.1).<br/>
   Paramètres: --ref-ld-chr /app/envs/ldscore/ldref/eur_w_ld_chr/ --w-ld-chr /app/envs/ldscore/ldref/eur_w_ld_chr/<br/><br/>
2. Manhattan plot résumant les associations SNPs (seuils: 5e-6 en vert, 5e-8 en rouge).<br/><br/>
3. QQ plot et lambda.<br/><br/>
4. Matrice de corrélation des SNPs significatifs (p-value < 5e-8), calculée via les corrélations entre les z-scores des dimensions latentes.<br/><br/>
5. Table des gènes (MAGMA v1.10, refpanel: g1000_eur, ensembl: v102, window: 35,10).<br/><br/>
   Au plus, les 20 gènes significativement associés ayant les plus faibles p-values sont présents dans la table.<br/><br/>
   Les gènes sont considérés comme associés de manière significative à la région lorsque p-value < 2e-6 (0.05/19264), après correction de Bonferroni  (19264 gènes).<br/><br/> 
6. Table des ensembles de gènes (MAGMA v1.10 avec MSigDB_20231Hs_MAGMA.txt).<br/><br/>
   Au plus, les 20 ensembles de gènes significativement associés ayant les plus faibles p-values sont présents dans la table.<br/><br/>
   Les ensembles de gènes sont considérés comme associés de manière significative à la région lorsque p-value < 2e-6 (0.05/17009), après correction de Bonferroni  (17009 ensembles.)

""".format(path_to_model= path_to_model,
    dim_formula="dim_i ~ C(Sex) + Age + I(Age*Age) + I(Age*Sex) + I(Age*Age*Sex) + C(Centre) + TICV + " + " + ".join(
        [f"PC{i:02d}" for i in range(1, nb_geneticPC + 1)]
    )
)

elements.append(Paragraph(methodology_text, style_normal))
elements.append(PageBreak())

# Initialize an empty list to store the heritability data
title_heritability = Paragraph("Maximum Heritability Estimates (h2) for Brain Regions", style_subtitle)
elements.append(title_heritability)
heritability_data = []

for region_model in regions_models:
    region, model, pca = region_model.split('/')
    base_path = os.path.expanduser(f"{path_to_Champollion}/{region}/{model}/{pca}/{population}")
    h2_path = os.path.join(base_path, "h2", "h2_summary.tsv")
    
    if os.path.exists(h2_path):
        h2_df = pd.read_csv(h2_path, sep="\t")
        if 'pheno' in h2_df.columns and 'h2' in h2_df.columns:
            # Extract the dimension with the highest heritability
            max_h2_row = h2_df.loc[h2_df['h2'].idxmax()]
            heritability_data.append([region, max_h2_row['pheno'], max_h2_row['h2']])

# Create a DataFrame for the heritability data
heritability_df = pd.DataFrame(heritability_data, columns=['Region', 'Most Heritable Dimension', 'h2'])
heritability_df = heritability_df.sort_values(by='h2', ascending=False)

heritability_table_data = [heritability_df.columns.tolist()] + heritability_df.values.tolist()
heritability_table = Table(heritability_table_data, colWidths=[200, 130, 80])
heritability_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 6),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
]))
elements.append(heritability_table)
elements.append(PageBreak())

title_top_shared_genes = Paragraph("Top 30 most shared genes across brain regions", style_title)
elements.append(title_top_shared_genes)
gene_counter = Counter()
bonferroni_threshold = 0.05 / 19264

for region_model in regions_models:
    region, model, pca = region_model.split('/')
    base_path = os.path.expanduser(f"{path_to_Champollion}/{region}/{model}/{pca}/{population}")
    magma_path = os.path.join(base_path, "MAGMA")
    magma_genes_file = os.path.join(magma_path, "magma.genes.out")
    
    if os.path.exists(magma_genes_file):
        print(f"Processing MAGMA genes for {region_model}...")
        genes_df = pd.read_csv(magma_genes_file, sep="\s+", comment="#")
        
        # Filter genes by the Bonferroni-corrected p-value threshold
        significant_genes = genes_df[genes_df["P"] < bonferroni_threshold]
        
        # Update the gene counter with the significant genes from this region
        gene_counter.update(significant_genes["GENE"])

# Convert the Counter to a DataFrame for easier manipulation
gene_counts_df = pd.DataFrame(gene_counter.items(), columns=["Gene", "Count"])

# Sort the DataFrame by the count in descending order and select the top 30
top_shared_genes = gene_counts_df.sort_values(by="Count", ascending=False).head(30)

genes_columns = list(top_shared_genes.columns)
top_shared_genes['Symbol'] = top_shared_genes['Gene'].apply(lambda x: get_gene_symbol(x))
top_shared_genes = top_shared_genes[['Symbol']+genes_columns]
data = [list(top_shared_genes.columns)] + top_shared_genes.values.tolist()

table = Table(data, colWidths=[50, 100, 35])
table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),  
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
elements.append(table)
elements.append(PageBreak())


title_top_loci = Paragraph("Top 30 genomic loci regarding the p-value across brain regions", style_title)
elements.append(title_top_loci)
top_loci_df = pd.DataFrame({'uniqID':[-9], 'rsID':[-9], 'chr':[-9],
            'start':[-9], 'end':[-9], 'p':[1], 'nGWASSNPs':[-9], 'Region':[-9]})
for region_model in regions_models:

    region, model, pca = region_model.split('/')
    base_path = os.path.expanduser(f"{path_to_Champollion}/{region}/{model}/{pca}/{population}")
    fuma_path = os.path.join(base_path, "FUMA")
    genomic_loci_file = os.path.join(fuma_path, "GenomicRiskLoci.txt")
    if os.path.exists(genomic_loci_file):
        loci_df = pd.read_csv(genomic_loci_file, sep="\t")
        loci_df = loci_df[['uniqID', 'rsID', 'chr',
            'start', 'end', 'p', 'nGWASSNPs']]
        loci_df["Region"] = region
        top_loci_df = pd.concat([top_loci_df, loci_df])
        top_loci_df = top_loci_df.sort_values(by="p", ascending=False)
        top_loci_df.drop_duplicates(subset='rsID',keep='last', inplace=True)
        top_loci_df = top_loci_df.iloc[-30:]


loci_columns = list(top_loci_df.columns)
top_loci_df = top_loci_df.sort_values(by="p", ascending=True)
top_loci_df['p'] = top_loci_df['p'].apply(lambda x: float(f"{x:.2g}"))
data = [list(top_loci_df.columns)] + top_loci_df.values.tolist()

table = Table(data, colWidths=[60, 55, 25, 45, 45, 30, 45, 90])
table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),  
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
elements.append(table)
elements.append(PageBreak())

# Loop through regions and models to add sections
for region_model in regions_models:
    region, model, pca = region_model.split('/')
    print(region)
    print(model)
    base_path = os.path.expanduser(f"{path_to_Champollion}/{region}/{model}/{pca}/{population}")
    magma_path = os.path.join(base_path, "MAGMA")
    
    # Title for each region-model
    title = f"{region} — {model}"
    title_paragraph = Paragraph(title, style_title)
    elements.append(title_paragraph)
    
    # h2 summary table
    h2_path = os.path.join(base_path, "h2", "h2_summary.tsv")
    if os.path.exists(h2_path):
        print("Processing h2...")
        h2_df = pd.read_csv(h2_path, sep="\t")
    
        if 'pheno' in h2_df.columns:
            h2_df['pheno_num'] = h2_df['pheno'].str.extract(r'dim(\d+)').astype(float)
            h2_df = h2_df.sort_values(by='pheno_num')
            h2_df = h2_df.drop(columns=['pheno_num'])  # optional: clean up
        
        data = [list(h2_df.columns)] + h2_df.values.tolist()

        table = Table(data, colWidths=[40, 40, 70, 70, 70])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),  # Smaller font size for all text
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elements.append(table)
        # Add a page break 
        elements.append(PageBreak())

    # Images (PNG files)
    for png_name in ["manhattan_plot.png"]:
        png_path = os.path.join(base_path, png_name)
        if os.path.exists(png_path):
            try:
                print(f"Processing PNG: {png_name}")
                # Create ReportLab image and scale
                img_reportlab = ReportLabImage(png_path)
                # Desired dimensions in points (1 point = 1/72 inch)
                img_reportlab.drawWidth = 500  # width in points
                img_reportlab.drawHeight = 281.25  # height in points

                elements.append(img_reportlab)
            except Exception as e:
                    print(f"[ERROR] Could not render PNG {png_path}: {e}")

    for png_name in ["QQplot.png"]:
        png_path = os.path.join(base_path, png_name)
        if os.path.exists(png_path):
            try:
                print(f"Processing PNG: {png_name}")
                # Create ReportLab image and scale
                img_reportlab = ReportLabImage(png_path)
                # Desired dimensions in points (1 point = 1/72 inch)
                img_reportlab.drawWidth = 200  # width in points
                img_reportlab.drawHeight = 200  # height in points

                elements.append(img_reportlab)
            except Exception as e:
                    print(f"[ERROR] Could not render PNG {png_path}: {e}")
    
    # Images (EPS files)
    for eps_name in ["Correlation_Matrix_SNPs_most.eps"]:
        eps_path = os.path.join(base_path, eps_name)
        if os.path.exists(eps_path):
            try:
                print("Processing .eps ...")
                # Open EPS using PIL and convert it to PNG
                img = Image.open(eps_path)
                img.load(scale=10) 
                img_width_px, img_height_px = img.size
                
                # Assume 96 DPI if not set (common for screen images)
                dpi = img.info.get('dpi', (96, 96))[0]
                px_to_pt = 72 / dpi
                # Convert pixels to points
                img_width_pt = img_width_px * px_to_pt
                img_height_pt = img_height_px * px_to_pt
            
                max_width = page_width - 100  # margins
                max_height = page_height - 100
                # Scale to fit A4 while maintaining aspect ratio
                scale = min(max_width / img_width_pt, max_height / img_height_pt, 1.0)
                final_width = img_width_pt * scale
                final_height = img_height_pt * scale

                # Save the image as a temporary PNG file
                img_path = f"/tmp/temp_image_{model}_{eps_name.replace('.eps', '')}.png" 
                img.save(img_path)

                # Create a ReportLab Image object
                img_reportlab = ReportLabImage(img_path, width=final_width, height=final_height)

                # Get the dimensions of the image
                img_width, img_height = img.size

                # Resize image to fit within the PDF page (with a max width)
                img_reportlab.width = img_width
                img_reportlab.height = img_height

                # Append the image to elements
                elements.append(img_reportlab)
                elements.append(PageBreak())
            except Exception as e:
                print(f"[ERROR] Could not render EPS {eps_name}: {e}")
    
    fuma_path = os.path.join(base_path, "FUMA")
    genomic_loci_file = os.path.join(fuma_path, "GenomicRiskLoci.txt")
    if os.path.exists(genomic_loci_file):
        print("Processing loci ...")
        loci_df = pd.read_csv(genomic_loci_file, sep="\t")
        cols = [
            'GenomicLocus', 'uniqID', 'rsID', 'chr',
            'start', 'end', 'p', 'nGWASSNPs'
            ]
        loci_df = loci_df[cols]
        loci_df['p'] = loci_df['p'].apply(lambda x: float(f"{x:.2g}"))
        data = [list(loci_df.columns)] + loci_df.values.tolist()

        table = Table(
            data,
            colWidths=[60, 60, 55, 25, 45, 45, 30, 45] 
        )

        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elements.append(table)
        elements.append(PageBreak())

    # MAGMA Genes Table
    magma_genes_file = os.path.join(magma_path, "magma.genes.out")
    if os.path.exists(magma_genes_file):
        print("Processing magma genes ...")
        genes_df = pd.read_csv(magma_genes_file, sep="\s+", comment="#")
        top_genes = genes_df.sort_values(by="P").head(20).drop(columns=["ZSTAT"], errors="ignore")
        top_genes = top_genes[top_genes["P"] < 0.05/19264]
        top_genes["P"] = top_genes["P"].apply(lambda x: float(f"{x:.2g}"))
        genes_columns = list(top_genes.columns)
        top_genes['Symbol'] = top_genes['GENE'].apply(lambda x: get_gene_symbol(x))
        top_genes = top_genes[['Symbol']+genes_columns]
        data = [list(top_genes.columns)] + top_genes.values.tolist()

        table = Table(data, colWidths=[50, 100, 25, 55, 55, 35, 40, 40, 70])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),  
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elements.append(table)
        # Add a page break 
        elements.append(PageBreak())
    
    # MAGMA Gene Sets Table
    magma_sets_file = os.path.join(magma_path, "magma.gsa.out")
    if os.path.exists(magma_sets_file):
        print("Processing magma gene sets ...")
        sets_df = pd.read_csv(magma_sets_file, sep="\s+", comment="#")
        top_sets = sets_df.sort_values(by="P").head(20).drop(columns=["VARIABLE", "TYPE"], errors="ignore")
        top_sets = top_sets[top_sets["P"] < 0.05/17009]
        top_sets["P"] = top_sets["P"].apply(lambda x: float(f"{x:.2g}"))
        data = [list(top_sets.columns)] + top_sets.values.tolist()

        table = Table(data, colWidths=[30, 50, 60, 60, 65, 250])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),  
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elements.append(table)
        elements.append(PageBreak())
    

# Build the PDF document
document.build(elements)

print(f"PDF written to: {output_file}")