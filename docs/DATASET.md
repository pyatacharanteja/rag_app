# Dataset

## Included sample documents (`sample_documents/`)
| File | Content | Pages |
|---|---|---|
| `northwind_energy_annual_report_2025.pdf` | Fictional renewable-energy company: revenue, segments, capex, sustainability | 3 |
| `contoso_retail_financial_report_2025.pdf` | Fictional retailer: revenue, margins, dividend, outlook | 3 |
| `fabrikam_plant_operations_manual.pdf` | Fictional plant manual: safety, maintenance, quality control | 3 |

**Source:** created for this project by `sample_documents/generate_samples.py`.
All company names, people and figures are invented, so there are no copyright or privacy issues and the
expected answers are known exactly (see `TEST_QUESTIONS.md`). The names Contoso, Northwind and Fabrikam are
generic placeholder names and do not refer to real company data.

## Using real documents (optional)
Any text-based PDF works. Suggested public sources:
- Annual / financial reports on the investor-relations pages of listed companies
- SEC EDGAR filings (10-K, 10-Q): <https://www.sec.gov/edgar/search/>
- Government or organisation technical reports and policy documents

Check the licence or terms of each document before redistributing it. Do not commit personal documents
(such as resumes) or confidential files to a public repository.

## Processing applied
Text is extracted per page with `pypdf`, whitespace is normalised, and pages are split into ~900-character
overlapping chunks. Each chunk stores `(text, file name, page number)`. Nothing is written to disk.
