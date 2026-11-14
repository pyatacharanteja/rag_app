"""Generates the three synthetic sample PDFs used for demos and tests.
All companies, people and figures are fictional (created for this project),
so there are no copyright or privacy concerns.
Run:  python sample_documents/generate_samples.py   (needs: pip install reportlab)
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).parent
S = getSampleStyleSheet()

DOCS = {
    "northwind_energy_annual_report_2025.pdf": [
        ("Northwind Energy - Annual Report 2025", [
            "Northwind Energy Ltd. is a renewable energy company founded in 2009 and headquartered in Pune, India. "
            "The company is led by Chief Executive Officer Anita Rao and employs 4,200 people.",
            "Financial highlights: Revenue for fiscal year 2025 was INR 8,450 crore, an increase of 12% over INR 7,545 crore "
            "in fiscal year 2024. Net profit for fiscal year 2025 was INR 910 crore.",
        ]),
        ("Business Segments and Capital Expenditure", [
            "Revenue by segment: Solar contributed 46%, Wind 31%, Energy Storage 14% and Services 9%. "
            "Total installed capacity reached 3.8 GW at the end of the year.",
            "Capital expenditure in fiscal year 2025 was INR 1,250 crore, mainly for new solar parks and battery storage.",
            "Key risks: changes in government policy, volatility in polysilicon supply, and grid connection delays.",
        ]),
        ("Sustainability", [
            "Northwind reduced its operational emissions by 18% compared with 2024 and has committed to net zero by 2040.",
            "The safety incident rate was 0.42 per million working hours, the lowest in the company's history.",
        ]),
    ],
    "contoso_retail_financial_report_2025.pdf": [
        ("Contoso Retail - Financial Report 2025", [
            "Contoso Retail operates 312 stores across 9 countries. Revenue for fiscal year 2025 was USD 2.1 billion, "
            "up 7% year over year. E-commerce accounted for 28% of total sales.",
            "Gross margin was 34.5% and operating income was USD 168 million.",
        ]),
        ("Governance, Dividends and Risks", [
            "The Chief Financial Officer is Daniel Mehta. The Board proposed a dividend of USD 0.45 per share.",
            "Principal risks include inflation in input costs, supplier concentration in two regions, and changing consumer demand.",
        ]),
        ("Outlook", [
            "For fiscal year 2026 management expects revenue growth of 6% to 8% and plans to open 25 new stores, "
            "while increasing investment in its e-commerce platform.",
        ]),
    ],
    "fabrikam_plant_operations_manual.pdf": [
        ("Fabrikam Plant Operations Manual - Safety", [
            "All employees must wear helmets and safety glasses in the production area. Fire drills are held every quarter. "
            "The emergency assembly point is Assembly Point B, next to the main gate.",
            "Any incident must be reported to the Shift Supervisor within 2 hours.",
        ]),
        ("Maintenance", [
            "Preventive maintenance is performed every 500 operating hours. Lubrication is carried out monthly and "
            "equipment calibration annually. The spare parts store is open from 7 am to 6 pm.",
        ]),
        ("Quality Control", [
            "The defect rate target is below 0.8%. Inspection samples 1 in every 50 units. "
            "Non-conforming material must be tagged with a red label and moved to the quarantine area.",
        ]),
    ],
}

for fname, pages in DOCS.items():
    story = []
    for i, (title, paras) in enumerate(pages):
        story.append(Paragraph(title, S["Heading1"]))
        for p in paras:
            story += [Paragraph(p, S["BodyText"]), Spacer(1, 10)]
        if i < len(pages) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(OUT / fname), pagesize=A4, title=fname).build(story)
    print("created", fname)
