#Transer Learning

#Instalamos el paquete de mxnet para redes neuronales

cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)
install.packages("mxnet",dependencies = T)
require(devtools)
install_version("DiagrammeR", version = "0.9.0", repos = "http://cran.us.r-project.org")
require(DiagrammeR)
library(mxnet)
library(imager)

#Descargamos la red inception, estado del arte en imagenet

download.file('http://data.dmlc.ml/data/Inception.zip', destfile = 'Inception.zip')
unzip("Inception.zip")
model <- mx.model.load("./Inception/Inception_BN", iteration = 39)

mean.img <- as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])

#Leemos las imagenes que queremos clasificar

im1 <- load.image("./Train/pistol/120px-JS05.jpg")
plot(im1)

im2 <- load.image("./Train/pistol/120px-K1200CloseUp.jpg")
plot(im2)

im3 <- load.image("./Train/smartphone/smartphone_113.jpg")
plot(im3)

im4 <- load.image("./Train/pistol/120px-Russian_Gas_Hand_Grenade_M1917.jpg")
plot(im4)



#Creamos una función de preprocesado

preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2)
  croped <- crop.borders(im, xx, yy)
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(croped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr) <- c(224, 224, 3)
  # subtract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}


#Normalizamos las imagenes

normed1 <- preproc.image(im1, mean.img)
normed2 <- preproc.image(im2, mean.img)
normed3 <- preproc.image(im3, mean.img)
normed4 <- preproc.image(im4, mean.img)

#Clasificamos las imagenes
synsets <- readLines("Inception/synset.txt")

prob <- predict(model, X = normed1)
max.idx <- max.col(t(prob))
print(paste0("Predicted Top-class: ", synsets[[max.idx]]))

prob <- predict(model, X = normed2)
max.idx <- max.col(t(prob))
print(paste0("Predicted Top-class: ", synsets[[max.idx]]))

prob <- predict(model, X = normed3)
max.idx <- max.col(t(prob))
print(paste0("Predicted Top-class: ", synsets[[max.idx]]))


prob <- predict(model, X = normed4)
max.idx <- max.col(t(prob))
print(paste0("Predicted Top-class: ", synsets[[max.idx]]))

