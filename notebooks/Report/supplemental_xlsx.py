import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

def populate_summary_sheet(ws):
    bold = Font(bold=True)
    title_font = Font(bold=True, size=14)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    section_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    wrap = Alignment(wrap_text=True, vertical="top")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    row = 1

    # ---------- 1. TITLE BLOCK ----------
    ws.cell(row=row, column=1, value="Supplementary Table: Champollion V1 association "
                                      "results").font = title_font
    row += 1
    ws.cell(row=row, column=1, value="Cohort: UK Biobank, 'white British' ancestry subset")
    row += 1
    ws.cell(row=row, column=1,
            value="Overview: SNP-based heritability (LDSC v2.0.1), genomic risk loci (FUMA 2.0.0), "
                  "gene- and gene-set-based association (MAGMA v1.10), developmental expression "
                  "enrichment (BrainSpan), and cross-trait genetic correlation, computed per "
                  "brain region and representations.")
    ws.cell(row=row, column=1).alignment = wrap
    ws.row_dimensions[row].height = 45
    row += 2  # blank row after title block

    # ---------- 2. TABLE OF CONTENTS ----------
    ws.cell(row=row, column=1, value="Table of contents").font = bold
    row += 1
    toc_headers = ["Sheet name", "Contents", "Key columns", "Filtering / threshold applied"]
    for j, h in enumerate(toc_headers, start=1):
        c = ws.cell(row=row, column=j, value=h)
        c.font = header_font
        c.fill = header_fill
        c.border = border
    row += 1

    toc_rows = [
        ("All region h2",
         "SNP-based heritability (LDSC v2.0.1) per region/residualised component",
         "region, h2, se, lambdaGC, Mean_Chi2, Intercept",
         "None (all rows included)"),
        ("All region loci",
         "Independent genomic loci from FUMA per region",
         "region, GenomicLocus, uniqID, rsID, chr, start, end, p, nGWASSNPs",
         "FUMA default significance threshold (p < 5e-8)"),
        ("All region top genes",
         "Top MAGMA gene-based association results per region",
         "region, Symbol, GENE, CHR, START, STOP, NSNPS, N, P",
         "Top 30 by P per region, then filtered to Bonferroni P < 0.05/19,264"),
        ("All region top sets",
         "Top MAGMA gene-set association results per region",
         "region, NGENES, BETA, BETA_STD, SE, P, FULL_NAME",
         "Top 30 by P per region, then filtered to Bonferroni P < 0.05/17,009"),
        ("All region Brainspan",
         "Developmental gene-expression enrichment (BrainSpan) per region",
         "region, VARIABLE, NGENES, BETA, BETA_STD, SE, P",
         "Top 20 by P per region, then filtered to Bonferroni P < 0.05/29"),
        ("All region gen corr",
         "Cross-trait genetic correlation (omnibus test) per region and disease",
         "region, trait, n_pcs, chi2, p_value, p_bonferroni, fdr_bh",
         "None (all rows included)"),
    ]
    for name, contents, cols, filt in toc_rows:
        for j, val in enumerate([name, contents, cols, filt], start=1):
            c = ws.cell(row=row, column=j, value=val)
            c.alignment = wrap
            c.border = border
        row += 1
    row += 1  # blank row

    # ---------- 3. COLUMN / ABBREVIATION GLOSSARY ----------
    ws.cell(row=row, column=1, value="Column and abbreviation glossary").font = bold
    row += 1
    glossary_headers = ["Term", "Definition"]
    for j, h in enumerate(glossary_headers, start=1):
        c = ws.cell(row=row, column=j, value=h)
        c.font = header_font
        c.fill = header_fill
        c.border = border
    row += 1

    glossary_rows = [
        ("h2", "SNP-based heritability estimate (LDSC)"),
        ("se", "Standard error of the estimate"),
        ("lambdaGC", "Genomic inflation factor"),
        ("Mean_Chi2 / Intercept", "LDSC diagnostics for residual confounding/inflation"),
        ("GenomicLocus / uniqID / rsID", "FUMA genomic risk locus identifiers"),
        ("chr / start / end", "Genomic coordinates (GRCh37) of the locus/gene"),
        ("nGWASSNPs", "Number of GWAS SNPs within the locus"),
        ("GENE", "Ensembl gene ID"),
        ("Symbol", "HGNC gene symbol"),
        ("NSNPS", "Number of SNPs in the MAGMA gene model"),
        ("N", "Sample size used in the MAGMA gene-based test"),
        ("BETA / BETA_STD", "Raw and standardized regression coefficients (MAGMA gene-set analysis)"),
        ("FULL_NAME", "Full name of the gene set tested"),
        ("VARIABLE", "BrainSpan developmental stage/time window tested"),
        ("n_pcs", "Number of principal components used for the phenotype"),
        ("chi2 / p_value", "Omnibus test statistic and associated p-value for genetic correlation"),
        ("p_bonferroni / fdr_bh", "Multiple-testing-corrected p-values (Bonferroni; Benjamini-Hochberg FDR)"),
    ]
    for term, definition in glossary_rows:
        c1 = ws.cell(row=row, column=1, value=term)
        c1.font = bold
        c1.border = border
        c2 = ws.cell(row=row, column=2, value=definition)
        c2.alignment = wrap
        c2.border = border
        row += 1
    row += 1  # blank row

    # ---------- 4. METHODS / SOFTWARE NOTES ----------
    ws.cell(row=row, column=1, value="Methods notes").font = bold
    row += 1
    methods_notes = [
        "Heritability and genetic correlation estimated using LDSC "
        "(specify version and reference LD panel used).",
        "Genomic risk loci identified using FUMA (specify version), "
        "significance threshold p < 5e-8 (folder label 'FUMA5e-08').",
        "Gene- and gene-set-based association performed using MAGMA "
        "(specify version and reference gene/gene-set definition files used).",
        "Developmental expression enrichment computed against the BrainSpan "
        "reference dataset (magma_exp_bs_age_avg_log2RPKM; 29 developmental "
        "stages/windows tested)."
    ]
    for note in methods_notes:
        c = ws.cell(row=row, column=1, value=f"\u2022 {note}")
        c.alignment = wrap
        row += 1
    row += 1  # blank row

    # ---------- 5. NOTES ON FILTERING ----------
    ws.cell(row=row, column=1, value="Notes on filtering").font = bold
    row += 1
    filtering_notes = [
        "'Top genes', 'top gene sets', and 'BrainSpan' sheets show the top N "
        "results par region by p-value (30, 30, and 20 respectively) AFTER filtering to "
        "the stated Bonferroni-corrected significance threshold — not simply "
        "the top N overall. As a result, some region combinations may "
        "show fewer than N rows, or none, if no results survived correction; "
        "this is expected and not a data truncation artifact.",
        "P-values throughout are rounded to 2 significant figures for display.",
        "Placeholder rows with value -9 in the 'region' columns "
        "(present in the initialized, empty DataFrames) should be disregarded "
        "if they appear; they indicate no data was available to append for "
        "that category before concatenation.",
    ]
    for note in filtering_notes:
        c = ws.cell(row=row, column=1, value=f"\u2022 {note}")
        c.alignment = wrap
        ws.row_dimensions[row].height = 45
        row += 1

    # ---------- Column widths ----------
    widths = {1: 28, 2: 45, 3: 45, 4: 45}
    for col, width in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = width

def create_sheet(df, title, drop_first_line=True):
    ws = wb.create_sheet(title)
    df = df.reset_index()
    if drop_first_line:
        df = df.drop(axis=0, index=0)
    df = df.drop('index', axis=1)
    n_row, n_col = df.shape
    for j in range(1,n_col+1):
        ws[f"{ALPHABET[j-1]}1"] = df.columns[j-1]
        for i in range(1,n_row+1):
            ws[f"{ALPHABET[j-1]}{i+1}"] = df.iloc[i-1,j-1]

def add_columns(df, region, model=None):
    df_cols = df.columns
    if region is not None and model is not None:
        df["region"] = region
        df["model"] = model
        df = df[["region", "model"]+list(df_cols)]
    elif region is not None:
        df["region"] = region
        df = df[["region"]+list(df_cols)]
    return df


path_to_Champollion = "/home/ad279118/ukb/26irene_AD_UKB_optim/results/Champollion_V1_32"
output_file = f"{path_to_Champollion}/Genetic_corr_supplemental.xlsx"
population = "White"
verbose = True
# Setup for the report
with open(f"{path_to_Champollion}/list_model_32PCs.txt") as f:
    regions_models = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

regions_models.sort()
#regions_models = regions_models[:2]
if verbose:
    print(regions_models)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

convertor_table = "/home/ad279118/ukb/ANNOT/martview_GRCh38.p14.txt"
convertor = pd.read_csv(
    convertor_table,
    sep=r"\s+",
    header=None,
    usecols=[0, 5],
    names=["ensembl", "SYMBOL"],
)

gene_dict = dict(zip(convertor["ensembl"], convertor["SYMBOL"]))

wb = Workbook()
ws = wb.active
ws.title = "Summary of the sheets"
populate_summary_sheet(ws)

all_h2_df = pd.DataFrame({'region':[-9], 'pheno':[-9],
            'h2':[-9], 'se':[-9], 'lambdaGC':[-9], 'Mean_Chi2':[-9], 'Intercept':[-9]}) #, 'model':[-9]

all_genomic_loci_df = pd.DataFrame({'region':[-9],
                'GenomicLocus':[-9], 'uniqID':[-9], 'rsID':[-9],
                'chr':[-9], 'start':[-9], 'end':[-9], 'p':[-9], 'nGWASSNPs':[-9]})  #, 'model':[-9]

all_top_gene_df = pd.DataFrame({'region':[-9], 
                'Symbol':[-9], 'GENE':[-9], 'CHR':[-9],
                'START':[-9], 'STOP':[-9], 'NSNPS':[-9], 'NPARAM':[-9], 'N':[-9], 'P':[-9]})  #, 'model':[-9]

all_top_gene_set_df = pd.DataFrame({'region':[-9],  
                'NGENES':[-9], 'BETA':[-9], 'BETA_STD':[-9],
                'SE':[-9], 'P':[-9], 'FULL_NAME':[-9]})  #, 'model':[-9]

all_brainspan_df = pd.DataFrame({'region':[-9],  
                'VARIABLE':[-9], 'NGENES':[-9], 'BETA':[-9],
                'BETA_STD':[-9], 'SE':[-9], 'P':[-9]})  #, 'model':[-9]

for region_model in regions_models:
    region, model, pca = region_model.split('/')
    if verbose:
        print(region)
        print(model)
    base_path = os.path.expanduser(f"{path_to_Champollion}/{region}/{model}/{pca}/{population}")
    magma_path = os.path.join(base_path, "MAGMA")
    
    # Title for each region-model
    title = f"{region}"# — {model}"
    
    # h2 summary table
    h2_path = os.path.join(base_path, "h2", "h2_summary.tsv")
    if os.path.exists(h2_path):
        if verbose:
            print("Processing h2...")
        h2_df = pd.read_csv(h2_path, sep="\t")
    
        if 'pheno' in h2_df.columns:
            h2_df['pheno_num'] = h2_df['pheno'].str.extract(r'dim(\d+)').astype(float)
            h2_df = h2_df.sort_values(by='pheno_num')
            h2_df = h2_df.drop(columns=['pheno_num']) 

            h2_df = add_columns(h2_df, region)
            all_h2_df = pd.concat([all_h2_df, h2_df])


    # Genomic loci
    fuma_path = os.path.join(base_path, "FUMA5e-08")
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

        loci_df = add_columns(loci_df, region)
        all_genomic_loci_df = pd.concat([all_genomic_loci_df, loci_df])

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

        top_genes = add_columns(top_genes, region)
        all_top_gene_df = pd.concat([all_top_gene_df, top_genes])
    
    all_top_gene_df = all_top_gene_df.drop("NPARAM", axis=1)

    magma_sets_file = os.path.join(magma_path, "magma.gsa.out")
    if os.path.exists(magma_sets_file):
        if verbose:
            print("Processing magma gene sets ...")
        sets_df = pd.read_csv(magma_sets_file, sep="\s+", comment="#")
        top_sets = sets_df.sort_values(by="P").head(30).drop(columns=["VARIABLE", "TYPE"], errors="ignore")
        top_sets = top_sets[top_sets["P"] < 0.05/17009]
        top_sets["P"] = top_sets["P"].apply(lambda x: float(f"{x:.2g}"))

        top_sets = add_columns(top_sets, region)    
        all_top_gene_set_df = pd.concat([all_top_gene_set_df, top_sets])

    BrainSpan_file = os.path.join(magma_path, "BrainSpan", "magma_exp_bs_age_avg_log2RPKM.gsa.out")
    if os.path.exists(BrainSpan_file):
        if verbose:
            print("Processing brainspan ...")
        BrainSpan_df = pd.read_csv(BrainSpan_file, sep="\s+", comment="#")
        top_BrainSpan = BrainSpan_df.sort_values(by="P").head(20).drop(columns=["TYPE"], errors="ignore")
        top_BrainSpan = top_BrainSpan[top_BrainSpan["P"] < 0.05/29]
        top_BrainSpan["P"] = top_BrainSpan["P"].apply(lambda x: float(f"{x:.2g}"))

        top_BrainSpan = add_columns(top_BrainSpan, region) 

        all_brainspan_df = pd.concat([all_brainspan_df, top_BrainSpan])

gen_corr_path = os.path.join(path_to_Champollion, "gencorr")
gen_corr_file = os.path.join(gen_corr_path, "all_omnibus.tsv")
if os.path.exists(gen_corr_file):
    if verbose:
        print("Processing gen corr...")
    gen_corr_df = pd.read_csv(gen_corr_file, sep="\t")
    cols = [
        'region', 'trait', 'n_pcs',
        'chi2', 'p_value', 'p_bonferroni', 'fdr_bh'
        ] #, 'model',
    gen_corr_df = gen_corr_df[cols]




create_sheet(all_h2_df, "All region h2")
create_sheet(all_genomic_loci_df, "All region loci")
create_sheet(all_top_gene_df, "All region top genes")
create_sheet(all_top_gene_set_df, "All region top sets")
create_sheet(all_brainspan_df, "All region Brainspan")
create_sheet(gen_corr_df, "All region gen corr", drop_first_line=False)

wb.save('/volatile/ad279118/trash/supplemental_material_tables.xlsx')