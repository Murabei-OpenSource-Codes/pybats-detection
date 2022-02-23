rm(list = ls())

rmarkdown::render(input = file.path("vignettes", "pybats_detection.Rmd"),
                  output_file = "pybats_detection",
                  output_format = rmarkdown::pdf_document(toc = TRUE),
                  clean = TRUE)

rmarkdown::render(input = file.path("vignettes", "quick_start.Rmd"),
                  output_file = "quick_start",
                  output_format = rmarkdown::html_document(toc_float = TRUE, toc = TRUE),
                     #rmarkdown::pdf_document(toc = TRUE, toc_depth = 4),
                  clean = TRUE)
