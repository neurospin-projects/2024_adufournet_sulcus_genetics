from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Table, TableStyle
from reportlab.platypus import Image as ReportLabImage
import os
import pandas as pd

path_to_ChampollionV1_256 = "."

# Setup for the report
with open("list_model.txt") as f:
    regions_models = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

"""[
    ("ScCal-SLi_left", "name07-41-43_180"),
    ("SFint-FCMant_left", "name08-09-20_81")
]"""

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
output_file = "Champollion_V1_256dim_noPCA_summary.pdf"

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

# Define a style for the content
style_normal = getSampleStyleSheet()["Normal"]

# Main report title
main_title = "Champollion V1 results in 256 dimensions, without PCA"
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
<b>Les modèles utilisés viennent de:</b><br/>
/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation_latent_256<br/>
(Entraînement sur 42,434 sujets).<br/><br/>

<b>Chacune des dimensions est résidualisée en utilisant la formule :</b><br/>
<font name="Courier">{dim_formula}</font><br/><br/>

<b>plink2-17-02-22 :</b><br/>
--snps-only<br/>
--maf 0.01<br/>
--max-alleles 2<br/>
--keep {{selected IID based on the stratification white.British.ancestry or non.white.British.ancestry}}<br/>
--geno<br/>
--mind<br/>
--hwe 1e-15<br/><br/>

Dans le cas de <b>white.British.ancestry</b>, 35,941 sujets sont détectés dans le bfile.<br/>
<b>Seuls les résultats issus de la cohorte de découverte</b> (white.British.ancestry) sont présents dans le PDF.<br/><br/>

Le nom de la région d'intérêt, suivie du modèle qui a été utilisé pour la représentation associée (par exemple: CINGULATE_left — name17-24-32_191)<br/><br/>

1. Tableau des résultats résumés de l'analyse LDSC (v2.0.1).<br/>
   Paramètres: --ref-ld-chr /app/envs/ldscore/ldref/eur_w_ld_chr/ --w-ld-chr /app/envs/ldscore/ldref/eur_w_ld_chr/<br/><br/>
2. Manhattan plot résumant les associations SNPs (seuils: 5e-6 en vert, 5e-8 en rouge).<br/><br/>
3. QQ plot et lambda.<br/><br/>
4. Matrice de corrélation des SNPs significatifs (p-value < 5e-8), calculée via les corrélations entre les z-scores des dimensions latentes.<br/><br/>
5. Table des gènes (MAGMA v1.10, refpanel: g1000_eur, ensembl: v102, window: 35,10).<br/><br/>
6. Table des ensembles de gènes (MAGMA v1.10 avec MSigDB_20231Hs_MAGMA.txt).
""".format(
    dim_formula="dim_i ~ Age +  C(Sex) + I(Age**2) + Age:C(Sex) + Cheadle + Newcastle + Array + " + " + ".join(
        [f"PC0{i}" if i < 10 else f"PC{i}" for i in range(1, 11)]
    )
)

elements.append(Paragraph(methodology_text, style_normal))
elements.append(PageBreak())

# Loop through regions and models to add sections
for region_model in regions_models:
    region, model = region_model.split('/')
    print(region)
    print(model)
    base_path = os.path.expanduser(f"~/tmp2/{region}/{model}/white.British.ancestry")
    magma_path = os.path.join(base_path, "MAGMA")
    
    # Title for each region-model
    title = f"{region} — {model}"
    title_paragraph = Paragraph(title, style_title)
    elements.append(title_paragraph)
    
    # h2 summary table
    h2_path = os.path.join(base_path, "h2_summary.tsv")
    if os.path.exists(h2_path):
        print("Processing h2...")
        h2_df = pd.read_csv(h2_path, sep="\t")
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
                img.load(scale=2) 
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
    
    # MAGMA Genes Table
    magma_genes_file = os.path.join(magma_path, "magma.genes.out")
    if os.path.exists(magma_genes_file):
        print("Processing magma genes ...")
        genes_df = pd.read_csv(magma_genes_file, sep="\s+", comment="#")
        top_genes = genes_df.sort_values(by="P").head(20).drop(columns=["ZSTAT"], errors="ignore")
        data = [list(top_genes.columns)] + top_genes.values.tolist()

        table = Table(data, colWidths=[120, 30, 70, 70, 40, 40, 40, 70])
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