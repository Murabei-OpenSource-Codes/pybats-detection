rm(list = ls())

rmarkdown::render(input = file.path("vignettes", "pybats_detection.Rmd"),
                  output_file = "pybats_detection",
                  # output_format = rmarkdown::html_document(toc_float = TRUE),
                  output_format = "latex_document",
                  # output_format = rmarkdown::pdf_document(toc = TRUE, toc_depth = 4),
                  clean = TRUE)

rmarkdown::render(input = file.path("vignettes", "quick_start.Rmd"),
                  output_file = "quick_start",
                  # output_format = rmarkdown::html_document(toc_float = TRUE, toc = TRUE),
                  output_format = rmarkdown::pdf_document(
                     toc = TRUE, toc_depth = 4, number_sections = TRUE,
                     dev = "pdf", fig_width = 16, fig_height = 6)
                  # , clean = TRUE
                  )
