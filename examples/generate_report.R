rm(list = ls())
path <- "./packages/pybats_detection/examples"
fname <- paste0(format(Sys.Date(), "%Y-%m-%d"), "__vignette")
rmarkdown::render(input = file.path(path, "index.Rmd"),
                  output_file = "vignette",
                  output_format = rmarkdown::pdf_document(toc = TRUE),
                  clean = TRUE)
