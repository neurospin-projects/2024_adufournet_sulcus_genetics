from PIL import Image
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Table, TableStyle
from reportlab.platypus import Image as ReportLabImage
from reportlab.lib.enums import TA_CENTER
from collections import Counter
import os
import pandas as pd

path_to_Champollion = "/home/ad279118/ukb/26irene_AD_UKB_optim/results/Champollion_V1_32"
Champollion_version = "32PCs_TIV_20genpcs"
nb_dim=  32
PCA = "with PCA" #reducting the latent space to 32 dimensions
folder = "32PCs" #32PCs
population="White"
path_to_model="/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation"
# "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation"
# "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation_latent_256"

path_variance = "/neurospin/dico/adufournet/2026_PCA_on_latent/ChampollionV1_32"
convertor_table = "/home/ad279118/ukb/ANNOT/martview_GRCh38.p14.txt"
nb_geneticPC=20

verbose = True
COUNT_FIGURE = 1
COUNT_TABLE = 1

convertor = pd.read_csv(
    convertor_table,
    sep=r"\s+",
    header=None,
    usecols=[0, 5],
    names=["ensembl", "SYMBOL"],
)

gene_dict = dict(zip(convertor["ensembl"], convertor["SYMBOL"]))

# Setup for the report
with open(f"{path_to_Champollion}/list_model_32PCs.txt") as f:
    regions_models = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

regions_models.sort()
#regions_models = regions_models[:2]
if verbose:
    print(regions_models)

class MyDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        # Automatically bookmark Paragraphs with style 'Title'
        if isinstance(flowable, Paragraph) and flowable.style.name == 'Title':
            key = flowable.getPlainText()
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(key, key, level=0, closed=False)

# Output PDF
output_file = f"{path_to_Champollion}/Champollion_V1_{Champollion_version}_supplemental.pdf"

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

style_caption = ParagraphStyle(
    "Caption",
    parent=getSampleStyleSheet()["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=9,
    leading=11,
    alignment=TA_CENTER,
    spaceBefore=6,
    spaceAfter=12,
    textColor=colors.black,
)

# Define a style for the content
style_normal = getSampleStyleSheet()["Normal"]

# Main report title
main_title = f"Supplemental Material: \n Champollion V1 association results"
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
<b>The used models are from:</b><br/>
{path_to_model}<br/>
(Trained on 42,433 subjects).<br/><br/>

<b>After the PCA, each of the principal component (dim) was pre-residualised using the formula:</b><br/>
<font name="Courier">{dim_formula}</font><br/><br/>

<b>plink2-17-02-22 :</b><br/>
--snps-only<br/>
--maf 0.01<br/>
--max-alleles 2<br/>
--keep {{selected IID based on white.British.ancestry}}<br/>
--geno<br/>
--mind<br/>
--hwe 1e-9<br/><br/>

In the case of <b>white.British.ancestry</b>, 35,940 subjects are detected in the bfile after the filters.<br/>
<b>Only the results from the discovery cohort</b> (white.British.ancestry) are presented in the PDF.<br/><br/>

The following are presented:<br/><br/>

Figure illustrating the extent to which the dimensions of the representation spaces (prior to any residualisation) are significantly correlated with the genetic principal components (PCs).<br/><br/>

Figure showing the distribution of LDSC regression intercepts across the univariate GWASs feeding into MOSTest.<br/><br/>

For each region, the following are presented: the region name, followed by the model that yielded the representation (for instance: CINGULATE_left — name17-24-32_191), and:<br/><br/>

    1. Figure of Manhattan plot summarizing SNP associations (thresholds: 5 × 10<super>−6</super> in green, 5 × 10<super>−8</super> in red).<br/><br/>
    2. Figure of QQ plot and lambda GC.<br/><br/>
    3. Figure of correlation matrix of significant SNPs (p-value < 5 × 10<super>−8</super>), calculated based on the correlations between the z-scores of the representation dimensions.<br/><br/>
    4. Figure of cell-type enrichment across developmental time points.<br/><br/>

""".format(path_to_model= path_to_model,
    dim_formula="dim_i ~ C(Sex) + Age + I(Age*Age) + I(Age*Sex) + I(Age*Age*Sex) + C(Centre) + TICV + " + " + ".join(
        [f"PC{i:02d}" for i in range(1, nb_geneticPC + 1)]
    )
)
#Table of the top 30 most shared genes across brain regions.<br/><br/>
#Table of the top 30 genomic loci regarding the p-value across brain regions.<br/><br/>
#    1. Table of heritability estimates computed using LDSC for each dimension of the representation space.<br/><br/>
#    5. Table of significant (5 × 10<super>−8</super>) genomic loci identified at LD threshold of r<super>2</super> = 0.1.<br/><br/>
#    6. Table of genes (MAGMA v1.10, refpanel: g1000_eur, ensembl: v102, window: ±20kb).<br/><br/>
#    At most, the 30 significantly associated genes with the lowest p-values are included in the table.<br/><br/>
#    Genes are considered to be significantly associated with the region when the p-value is less than 2 × 10<super>−6</super> (0.05/19,264), after Bonferroni correction (19,264 genes).<br/><br/>   
#    7. Table of gene sets (MAGMA v1.10 with MSigDB_20231Hs_MAGMA.txt).<br/><br/>
#    At most, the 30 significantly associated gene sets with the lowest p-values are included in the table.<br/><br/>
#    Gene sets are considered to be significantly associated with the region when the p-value is less than 2 × 10<super>−6</super> (0.05/17,009), after Bonferroni correction (17,009 gene sets).<br/><br/>
#    8. Table of BrainSpan enrichment analysis.<br/><br/>

elements.append(Paragraph(methodology_text, style_normal))
elements.append(PageBreak())

base_path = "/neurospin/dico/adufournet/2026_Nature/images"
png_name = "white_pvalues_distribution.png"
png_path = os.path.join(base_path, png_name)
if os.path.exists(png_path):
    try:
        if verbose:
            print(f"Processing PNG: {png_name}")
        # Create ReportLab image and scale
        img_reportlab = ReportLabImage(png_path)
        # Desired dimensions in points (1 point = 1/72 inch)
        img_reportlab.drawWidth = 277*1.25  # width in points
        img_reportlab.drawHeight = 444*1.25  # height in points

        elements.append(img_reportlab)
        caption = (
            f"<b>Figure {COUNT_FIGURE}:</b> For each dimension of the 32-dimensional representation space (after the pca), "
            f"across all 56 regions, we regressed subject-level coefficients (35,940 white British ancestry "
            f"participants) against each of the 40 leading genetic principal components, yielding 32×56 = 1,792 "
            f"p-values per PC. This tests whether the self-supervised representations (trained without "
            f"genetic information) nonetheless carry structure aligned with population genetic axes."
            f"The resulting p-value distributions are close to uniform for the majority of PCs. "
            f"Seven PCs (PC4, 5, 9, 10, 11, 12, 14) show a modest "
            f"excess of low p-values relative to the uniform expectation. All seven fall within "
            f"the first 20 genetic PCs already included as covariates in the pre-residualisation "
            f"step of the main mGWAS (Methods 5.8), and are therefore regressed out of the multivariate phenotypes "
            f"prior to association testing."
        )

        elements.append(Paragraph(caption, style_caption))
        elements.append(PageBreak())
        COUNT_FIGURE +=1
    except Exception as e:
            print(f"[ERROR] Could not render PNG {png_path}: {e}")


png_name = "Intercept_distribution.png"
png_path = os.path.join(base_path, png_name)
if os.path.exists(png_path):
    try:
        if verbose:
            print(f"Processing PNG: {png_name}")
        # Create ReportLab image and scale
        img_reportlab = ReportLabImage(png_path)
        # Desired dimensions in points (1 point = 1/72 inch)
        img_reportlab.drawWidth = 500  # width in points
        img_reportlab.drawHeight = 333  # height in points

        elements.append(img_reportlab)
        caption = (
            f"<b>Figure {COUNT_FIGURE}:</b> Distribution of LDSC regression intercepts across the univariate GWASs feeding into MOSTest "
            f"(i.e., one intercept per representation dimension × region, ~32×56 = 1,792 tests). "
            f"LDSC intercepts near 1.0 indicate the inflation in test statistics is attributable to polygenicity rather than confounding. "
        )

        elements.append(Paragraph(caption, style_caption))
        elements.append(PageBreak())
        COUNT_FIGURE +=1
    except Exception as e:
            print(f"[ERROR] Could not render PNG {png_path}: {e}")

# Table of LDSC (v2.0.1) aggregated heritability.<br/>
# The aggregated heritability is computed as a variance-weighted sum of per-dimension heritabilities (see Methods).<br/>
# Parameters: --ref-ld-chr /app/envs/ldscore/ldref/eur_w_ld_chr/ --w-ld-chr /app/envs/ldscore/ldref/eur_w_ld_chr/<br/><br/>

# title_global_heritability = Paragraph("Aggregated Heritability Estimates (h2) for each representation space", style_subtitle)
# elements.append(title_global_heritability)

# dic_global_h2 = {"region":[], 
#                "model":[],
#                "h2":[], 
#                "se":[],
#                }

# for region_model in regions_models:
#     region, model, pca = region_model.split('/')
#     base_path = os.path.expanduser(f"{path_to_Champollion}/{region}/{model}/{pca}/{population}")
#     h2_path = os.path.join(base_path, "h2", "h2_summary.tsv")

#     var_path = os.path.join(path_variance, f"{region}_explained_variance_ratio.csv")
#     if verbose : 
#         print(var_path)
#     if os.path.exists(var_path):
#         var_df = pd.read_csv(var_path)
#         var_df["pheno"] = var_df["pheno"].apply(lambda x : x+"_res")
    
#     if os.path.exists(h2_path):
#         h2_df = pd.read_csv(h2_path, sep="\t")

#     if os.path.exists(var_path) and os.path.exists(h2_path):
#         merge = pd.merge(var_df, h2_df, on='pheno')

#         evr = np.array(merge["explained_variance_ratio"])
#         h2  = np.array(merge["h2"])
#         se  = np.array(merge["se"])

#         if verbose:
#             print(f"Sum of EVR (should be ~1): {evr.sum():.6f}")

#         # Weighted h²
#         h2_region = np.sum(evr * h2)

#         # Propagated SE: sqrt(sum of (w_k * se_k)^2) — assuming independence across PCs
#         se_region = np.sqrt(np.sum((evr * se)**2))

#         if verbose:
#             print(f"\nWeighted h²_region : {h2_region:.4f}")
#             print(f"Propagated SE      : {se_region:.4f}")
#             print(f"95% CI             : [{h2_region - 1.96*se_region:.4f}, {h2_region + 1.96*se_region:.4f}]")

#             # Show contribution of each PC to final estimate
#             print("\nPer-PC contributions (w_k * h²_k):")
#             contributions = evr * h2
#             for i, (w, h, c) in enumerate(zip(evr, h2, contributions)):
#                 print(f"  dim{i+1:02d}: w={w:.4f}  h²={h:.4f}  contribution={c:.4f}  ({100*c/h2_region:.1f}%)")

#         dic_global_h2["region"].append(region)
#         dic_global_h2["model"].append(model)
#         dic_global_h2["h2"].append(h2_region)
#         dic_global_h2["se"].append(se_region)
    
#     else : 
#         print(region, model, "not found")

# df_global_h2 = pd.DataFrame.from_dict(dic_global_h2)
# df_global_h2 = df_global_h2[["region", "h2", "se"]].sort_values(by="h2", ascending=False)
# df_global_h2["h2"] = df_global_h2["h2"].apply(lambda x: f"{x:.3f}")
# df_global_h2["se"] = df_global_h2["se"].apply(lambda x: f"{x:.3f}")

# global_h2_table_data = [df_global_h2.columns.tolist()] + df_global_h2.values.tolist()
# global_h2_table = Table(global_h2_table_data, colWidths=[200, 130, 80])
# global_h2_table.setStyle(TableStyle([
#     ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#     ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#     ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#     ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#     ('FONTSIZE', (0, 0), (-1, -1), 6),
#     ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#     ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
# ]))
# elements.append(global_h2_table)
# caption = (
#     f"<b>Table {COUNT_TABLE}:</b> SNP-based h2 aggregated for each regional "
#     f"representation space."
# )
# elements.append(Paragraph(caption, style_caption))
# COUNT_TABLE+=1
# elements.append(PageBreak())


"""title_top_shared_genes = Paragraph("Top 30 most shared genes across brain regions", style_title)
elements.append(title_top_shared_genes)
gene_counter = Counter()
bonferroni_threshold = 0.05 / 19264

for region_model in regions_models:
    region, model, pca = region_model.split('/')
    base_path = os.path.expanduser(f"{path_to_Champollion}/{region}/{model}/{pca}/{population}")
    magma_path = os.path.join(base_path, "MAGMA")
    magma_genes_file = os.path.join(magma_path, "magma.genes.out")
    
    if os.path.exists(magma_genes_file):
        if verbose:
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
top_shared_genes['Symbol'] = top_shared_genes['Gene'].apply(lambda x: gene_dict.get(x, None))
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
caption = (
    f"<b>Table {COUNT_TABLE}:</b> Top 30 most shared genes across brain regions. A gene must be considered as "
    f"significant (p-value below {bonferroni_threshold:.2g}) to be counted for one region."
)
elements.append(Paragraph(caption, style_caption))
COUNT_TABLE+=1
elements.append(PageBreak())
if verbose:
    print("Top 30 most shared genes across brain regions, DONE")"""


"""title_top_loci = Paragraph("Top 30 genomic loci regarding the p-value across brain regions", style_title)
elements.append(title_top_loci)
top_loci_df = pd.DataFrame({'uniqID':[-9], 'rsID':[-9], 'chr':[-9],
            'start':[-9], 'end':[-9], 'p':[1], 'nGWASSNPs':[-9], 'Region':[-9]})
for region_model in regions_models:

    region, model, pca = region_model.split('/')
    base_path = os.path.expanduser(f"{path_to_Champollion}/{region}/{model}/{pca}/{population}")
    fuma_path = os.path.join(base_path, "FUMA5e-08")
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
    else :
        print(genomic_loci_file, "doesn't exist.")


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
caption = (
    f"<b>Table {COUNT_TABLE}:</b> Top 30 genomic loci regarding the p-value of the lead SNP representing the loci, across brain regions."
)
elements.append(Paragraph(caption, style_caption))
COUNT_TABLE+=1
elements.append(PageBreak())
if verbose:
    print("Top 30 genomic loci regarding the p-value across brain regions, DONE")"""

# Loop through regions and models to add sections
for region_model in regions_models:
    region, model, pca = region_model.split('/')
    if verbose:
        print(region)
        print(model)
    base_path = os.path.expanduser(f"{path_to_Champollion}/{region}/{model}/{pca}/{population}")
    magma_path = os.path.join(base_path, "MAGMA")
    
    # Title for each region-model
    title = f"{region} — {model}"
    title_paragraph = Paragraph(title, style_title)
    elements.append(title_paragraph)
    
    """# h2 summary table
    h2_path = os.path.join(base_path, "h2", "h2_summary.tsv")
    if os.path.exists(h2_path):
        if verbose:
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
        caption = (
        f"<b>Table {COUNT_TABLE}:</b> Heritability estimates computed using LDSC "
        f"for each dimension of the representation space encoding the <b>{region}</b> region. "
        f"The reported metrics include SNP-based heritability (h²), standard error "
        f"(SE), genomic inflation factor (λGC), mean χ² statistic, and LDSC intercept."
                )
        elements.append(Paragraph(caption, style_caption))
        COUNT_TABLE+=1
        elements.append(PageBreak())"""

    # Images (PNG files)
    for png_name in ["manhattan_plot.png"]:
        png_path = os.path.join(base_path, png_name)
        if os.path.exists(png_path):
            try:
                if verbose:
                    print(f"Processing PNG: {png_name}")
                # Create ReportLab image and scale
                img_reportlab = ReportLabImage(png_path)
                # Desired dimensions in points (1 point = 1/72 inch)
                img_reportlab.drawWidth = 500  # width in points
                img_reportlab.drawHeight = 281.25  # height in points

                elements.append(img_reportlab)
                caption = (
                    f"<b>Figure {COUNT_FIGURE}:</b> Manhattan plot showing significance of each variant’s association (using MOSTest) "
                    f"with the <b>{region}</b> region encoded by Champollion V1 in "
                    f"35,940 UK Biobank participants of white British ancestry. "
                    f"Each blue point represents a SNP, with lead SNPs highlighted in red. "
                    f"The conventional genome-wide significance threshold "
                    f"(5 × 10<super>−8</super>) is indicated by the red horizontal line."
                    f"The green line marks the SNP suggestive threshold (5 × 10<super>−6</super>)."
                )

                elements.append(Paragraph(caption, style_caption))
                elements.append(PageBreak())
                COUNT_FIGURE +=1
            except Exception as e:
                    print(f"[ERROR] Could not render PNG {png_path}: {e}")

    for png_name in ["QQplot.png"]:
        png_path = os.path.join(base_path, png_name)
        if os.path.exists(png_path):
            try:
                if verbose:
                    print(f"Processing PNG: {png_name}")
                # Create ReportLab image and scale
                img_reportlab = ReportLabImage(png_path)
                # Desired dimensions in points (1 point = 1/72 inch)
                img_reportlab.drawWidth = 200  # width in points
                img_reportlab.drawHeight = 200  # height in points

                elements.append(img_reportlab)
                caption = (
                    f"<b>Figure {COUNT_FIGURE}:</b> SNP based QQ plot and genomic inflation factors "
                    f"for the <b>{region}</b> region (based on the MOSTest associations results on the represention)."
                )

                elements.append(Paragraph(caption, style_caption))
                elements.append(PageBreak())
                COUNT_FIGURE +=1
            except Exception as e:
                    print(f"[ERROR] Could not render PNG {png_path}: {e}")
    
    # Images (EPS files)
    for eps_name in ["Correlation_Matrix_SNPs_most.eps"]:
        eps_path = os.path.join(base_path, eps_name)
        if os.path.exists(eps_path):
            try:
                if verbose:
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
                caption = (
                    f"<b>Figure {COUNT_FIGURE}:</b> Pairwise correlation matrix of multivariate "
                    f"SNP association profiles for genome-wide significant SNPs (p<5 × 10<super>−8</super>) in the "
                    f"<b>{region}</b> region. Each row/column represents one SNP, "
                    f"characterized by its vector of 32 univariate GWAS z-scores (one "
                    f"per representation dimension). Cell colour indicates the Pearson correlation "
                    f"between two SNPs' z-score vectors. The block-diagonal structure (red) reflects SNPs "
                    f"in high linkage disequilibrium at the same locus, which (as expected) show near- "
                    f"identical association profiles across the 32 representation dimensions, confirming "
                    f"that the multivariate signal is driven by coherent, locus-level genetic effects rather "
                    f"than dimension-specific noise. Negative correlations (blue) inside "
                    f"some blocks reflect that regressions were performed with respect to minor allele "
                    f"frequency rather than a fixed strand, so a SNP coded on the complementary strand relative to the majority of the SNPs within a block"
                    f"yields a z-score vector of equal magnitude and opposite sign."
                )

                elements.append(Paragraph(caption, style_caption))
                elements.append(PageBreak())
                COUNT_FIGURE +=1
            except Exception as e:
                print(f"[ERROR] Could not render EPS {eps_name}: {e}")
    
    """fuma_path = os.path.join(base_path, "FUMA5e-08")
    genomic_loci_file = os.path.join(fuma_path, "GenomicRiskLoci.txt")
    if os.path.exists(genomic_loci_file):
        if verbose:
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
        caption = (
        f"<b>Table {COUNT_TABLE}:</b> Genomic loci identified for the <b>{region}</b> region, "
        f"after the second clumping procedure (Methods), using a more "
        f"stringent LD threshold of r2 = 0.1. Clumps separated by less "
        f"than 250 kb were subsequently merged into a single genomic locus, represented by the lead SNP "
        f"with the lowest p-value. "
                )
        elements.append(Paragraph(caption, style_caption))
        COUNT_TABLE+=1
        elements.append(PageBreak())"""

    """# MAGMA Genes Table
    magma_genes_file = os.path.join(magma_path, "magma.genes.out")
    if os.path.exists(magma_genes_file):
        if verbose:
            print("Processing magma genes ...")
        genes_df = pd.read_csv(magma_genes_file, sep="\s+", comment="#")
        top_genes = genes_df.sort_values(by="P").head(30).drop(columns=["ZSTAT"], errors="ignore")
        top_genes = top_genes[top_genes["P"] < 0.05/19264]
        top_genes["P"] = top_genes["P"].apply(lambda x: float(f"{x:.2g}"))
        genes_columns = list(top_genes.columns)
        top_genes['Symbol'] = top_genes['GENE'].apply(lambda x: gene_dict.get(x, None))
        top_genes = top_genes[['Symbol']+genes_columns]
        data = [list(top_genes.columns)] + top_genes.values.tolist()
        COUNT_GENE = len(top_genes.values.tolist())

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
        caption = (
        f"<b>Table {COUNT_TABLE}:</b> Top 30 genes identified for the <b>{region}</b> region, "
        f"with the MAGMA software (Methods). "
        f"Only genes with an enrichment p-value below (0.05/19264)={(0.05/19264):.1g} are presented."
                )
        elements.append(Paragraph(caption, style_caption))
        COUNT_TABLE+=1
        elements.append(PageBreak())"""
    
    """# MAGMA Gene Sets Table
    magma_sets_file = os.path.join(magma_path, "magma.gsa.out")
    if os.path.exists(magma_sets_file):
        if verbose:
            print("Processing magma gene sets ...")
        sets_df = pd.read_csv(magma_sets_file, sep="\s+", comment="#")
        top_sets = sets_df.sort_values(by="P").head(30).drop(columns=["VARIABLE", "TYPE"], errors="ignore")
        top_sets = top_sets[top_sets["P"] < 0.05/17009]
        top_sets["P"] = top_sets["P"].apply(lambda x: float(f"{x:.2g}"))
        data = [list(top_sets.columns)] + top_sets.values.tolist()
        COUNT_GENE_SET = len(top_sets.values.tolist())

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
        caption = (
        f"<b>Table {COUNT_TABLE}:</b> Top 30 gene sets identified for the <b>{region}</b> region, "
        f"with the MAGMA software (Methods). "
        f"Only gene sets with an enrichment p-value below (0.05/17009)={(0.05/17009):.1g} are presented."
                )
        elements.append(Paragraph(caption, style_caption))
        COUNT_TABLE+=1
        elements.append(PageBreak())"""

    """BrainSpan_file = os.path.join(magma_path, "BrainSpan", "magma_exp_bs_age_avg_log2RPKM.gsa.out")
    if os.path.exists(BrainSpan_file):
        if verbose:
            print("Processing brainspan ...")
        BrainSpan_df = pd.read_csv(BrainSpan_file, sep="\s+", comment="#")
        top_BrainSpan = BrainSpan_df.sort_values(by="P").head(20).drop(columns=["TYPE"], errors="ignore")
        top_BrainSpan = top_BrainSpan[top_BrainSpan["P"] < 0.05/29]
        top_BrainSpan["P"] = top_BrainSpan["P"].apply(lambda x: float(f"{x:.2g}"))
        data = [list(top_BrainSpan.columns)] + top_BrainSpan.values.tolist()

        table = Table(data, colWidths=[35, 50, 60, 60, 65, 50])
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
        caption = (
        f"<b>Table {COUNT_TABLE}:</b> BrainSpan enrichment analysis for the <b>{region}</b> region, "
        f"with the MAGMA software (Methods). "
        f"Only time points with an enrichment p-value below (0.05/29)={(0.05/29):.1g} are presented."
                )
        elements.append(Paragraph(caption, style_caption))
        COUNT_TABLE+=1
        elements.append(PageBreak())"""

    
    singlecell_file = os.path.join(magma_path, "SINGLECELL", "plot_log10FDR_level1.png")
    if os.path.exists(singlecell_file):
        try:
            if verbose:
                print(f"Processing PNG: SINGLECELL")
            # Create ReportLab image and scale
            img_reportlab = ReportLabImage(singlecell_file)
            # Desired dimensions in points (1 point = 1/72 inch)
            img_reportlab.drawWidth = 451  # width in points
            img_reportlab.drawHeight = 208  # height in points

            elements.append(img_reportlab)
            caption = (
                    f"<b>Figure {COUNT_FIGURE}:</b> Cell-type enrichment across developmental time points"
                    f" for the "
                    f"<b>{region}</b> region. "
                    f"Each point represents the enrichment "
                    f"significance of a given cell type obtained from the MAGMA gene-property "
                    f"analysis using fetal cortical single-cell RNA-sequencing data. The y-axis " 
                    f"shows the FDR-corrected significance (-log10(FDR)), where larger values "
                    f"indicate stronger evidence for enrichment after Benjamini-Hochberg correction "
                    f"across the cell types tested within each developmental time point. "
                    f"The -log10(0.05) threshold is indicated with the red line. "
                    f"Point colours denote the developmental stage (post-conception weeks, PCW), "
                    f"allowing visualisation of temporal changes in the cell-type enrichment profile throughout cortical development."
                )

            elements.append(Paragraph(caption, style_caption))
            elements.append(PageBreak())
            COUNT_FIGURE+=1
        except Exception as e:
                print(f"[ERROR] Could not render PNG {singlecell_file}: {e}")
    

# Build the PDF document
document.build(elements)

print(f"PDF written to: {output_file}")