# memanggil package yang diperlukan 
library(raster)
library(terra) # digunakan untuk membaca data raster dan vektor 
library(sf) # simple feature mengolah data vektor (shapefile, GeoJSON)
library(neuralnet) # Membuat model jaringan saraf tiruan (BPNN)
library(ggplot2) # Membuat visualisasi data grafik, chart (sintaksis grammar of graphics)
library(lattice) # Membuat visualisasi data grafik, chart yang lebih rumit (sintaksis grid graphics) 
library(caret) # Membuat model machine learning

# Memanggil dan memmbaca data
sentinel_data <- rast("data/Sentinel_2024.tif") # Sintaks 'rast' Memanggil data raster (tif)
training <- st_read("data/koleksi.shp") # Sintaks 'st_read' Memanggil data vector (shp)

# Visualisasi Data
plotRGB(sentinel_data, r = 3, g = 2, b = 1, stretch = "lin", main = 'Citra Sentinel dengan Titik Sampling') # mengatur data raster yang dipanggil pada window plot
plot(st_geometry(training), add = TRUE, col = "red", pch = 5) # mengatur data vector yang dipanggil pada window plot

# preprocessing dengan memeriksa struktur awal
print(head(training)) # memunculkan tabel atribute train sampel data vector pada console
print(unique(training$LU)) # memunculkan simbol unik pada tabel atribut yang akan dipakai (value)

# Memastikan Field Id adalah faktor/numerik berurutan
training$LU <- as.factor(training$LU)

# Transformasi proyeksi
trainarea_PCS <- st_transform(training, crs(sentinel_data))

# Ekstrak nilai spektral dari titik sampel pada citra
nilai_sampel <- extract(sentinel_data, training) # ekstrak data atau mengambil bagian dari data raster ke sampel (8 band dengan train_sampel) 

# Gabungkan dengan atribut sampel
sampel_data <- cbind(st_drop_geometry(training), nilai_sampel) # menghapus train_sampel dan digabungkan oleh nilai_sampel dengan membuat kolom baru
sampel_data <- sampel_data[, c("LU", names(sentinel_data))] # menggabungkan sampel data dengan data sentinel

# Hapus data dengan nilai NA
sampel_data <- na.omit(sampel_data) # menghapus baris dengan nilai NULL(NA) dari data frame

# Debug: Periksa struktur setelah preprocessing
print(head(sampel_data)) # # memunculkan tabel atribute sampel_data pada console

# memulai Split data menjadi pembelajaran (training) dan pengujian (testing)
set.seed(43) # mengatur seed generator bilangan acak, memastikan hasil acak yang sama saat menjalankan kode berulang kali

# Pembagian data menjadi pelatihan (70%) dan pengujian (30%), 
index <- createDataPartition(sampel_data$LU, p = 0.7, list = FALSE) # memisahkan atau membagi data index pembelajaran (training) dan pengujian (testing)
train_data <- sampel_data[index, ] # memasukkan nilai sampel_data dengan index yang sudah dibuat menjadi train_data
test_data <- sampel_data[-index, ] # memasukkan nilai sampel_data dengan -index yang sudah dibuat menjadi test_data

# Normalisasi data
preProcess_params <- preProcess(train_data[, -1], method = c("center", "scale")) # perepocess data yaitu menyiapkan data yang dipakai pada train_data
train_input <- predict(preProcess_params, train_data[, -1]) # memprediksi nilai dari model statistik train_data yang telah dibuat
test_input <- predict(preProcess_params, test_data[, -1]) # memprediksi nilai dari model statistik test_data yang telah dibuat

# Tambahkan label ke data training
train <- cbind(train_input, LU = train_data$LU) # menggabungkan dua data frame (train_input dan LU) secara horizontal (menjadi kolom baru)

# Konversi label LU ke faktor untuk multi-class
train$LU <- as.factor(train$LU) # mengubah kolom data LU menjadi faktor yang memiliki nilai kategori atau label

# Definisikan formula untuk neural network
formula <- as.formula(paste("LU ~", paste(names(train_input), collapse = " + "))) # "as.formula" membuat formula atau representasi matematika dari model statistik
                                                                                  # "collapse" menggabungkan atau mengurangi dimensi data
# memasukkan model klasifikasi menggunakan neural network pada sintaks 'neuralnet', dengan hidden layer 10, dan data 'train'
nn_model <- neuralnet(
  formula,
  data = train, # variabel data yang dimasukkan dan dikeluarkan
  hidden = c(10, 5), # jumlah hidden layer yang dimasukkan, untuk dua lapisan  
  linear.output = FALSE, # menentukan apakah output model linear (TRUE) atau tidak (FALSE) 
  act.fct = "logistic", # fungsi aktivasi untuk neuron tersembunyi untuk mempelajari dan menggeneralisasi data, pada "logistic" digunakan untuk binary classification atau model probabilitas
  stepmax = 1e6 # fungsi aktiviasi step pada jangkauan maximum
) 

plot(nn_model, main = "Arsitektur Neural Network")

# Prediksi pada data testing
prediksi <- compute(nn_model, test_input) # sintaks "compute" untuk menghitung nilai dari fungsi
predicted_classes <- apply(prediksi$net.result, 1, which.max) - 1  # sintaks "apply" untuk mengaplikasikan fungsi dan -1 untuk menyesuaikan indeks kelas

# Evaluasi hasil
test_output <- test_data$LU # untuk mengalokasikan data pada nilai kolom LU dari dataframe test_data
confusion_matrix <- table(Predicted = predicted_classes, Actual =  as.numeric(test_output)) # membuat confussion matrix
print(confusion_matrix) # memunculkan nilai confussion matrix pada console

akurasi <- sum(diag(confusion_matrix)) / sum(confusion_matrix) # menghitung akurasi
print(paste("Akurasi:", round(akurasi * 100, 2), "%")) # memunculkan nilai akurasi pada console 

# Prediksi pada seluruh citra dengan Konversi raster ke data frame
sentinel_data_df <- as.data.frame(sentinel_data, xy = FALSE) # mengubah struktur data sentinel_data menjadi data frame yaitu baris dan kolom
colnames(sentinel_data_df) <- names(sentinel_data)  # Tetapkan nama kolom
sentinel_data_matrix <- as.matrix(sentinel_data_df) # mengubah data menjadi matrix

# memeriksa nilai null pada data agar terbaca
valid_mask <- !is.na(sentinel_data_matrix[, 1]) # memeriksa data apakah ada yang NA (null), yang akan terbaca TRUE pada operator "!" digunakan untuk membalikkan hasil   
sentinel_data_matrix[!valid_mask, ] <- 0 # memeriksa data dari valid_mask sebelumnya

# Normalisasi data citra
sentinel_data_matrix_norm <- as.data.frame(sentinel_data_matrix) # membalikkan data atau menormalisasi data sentinel_data_matrix
sentinel_data_matrix_norm <- predict(preProcess_params, sentinel_data_matrix_norm) # prediksi hasil sentinel_data_matrix_norm

# Prediksi menggunakan model neural network
prediksi_raster <- compute(nn_model, sentinel_data_matrix_norm) # sintaks "compute" untuk menghitung nilai dari fungsi
predicted_classes_raster <- as.integer(apply(prediksi_raster$net.result, 1, which.max)) - 1 # mengubah nilai menjadi tipe data integral (integer)

# Ubah ke raster
predicted_raster <- rast(sentinel_data) # mengeluarkan hasil prediksi NN menjadi raster
values(predicted_raster) <- ifelse(valid_mask, predicted_classes_raster, NA) # sintaks "ifelse" untuk melakukan operasi kondisional, yaitu memilih antara dua nilai berdasarkan kondisi yang ditentukan

# Visualisasi hasil
kelas_warna <- c("blue", "red", "gray", "darkgreen") # mengatur visualisasi warna 
kelas_label <- c("Badan Air", "Lahan Terbangun", "Lahan Terbuka", "Vegetasi") # mengatur visualisasi kelas

#memunculkan visuaslisai pada plot
plot(predicted_raster[[1]], 
     main = "Hasil Klasifikasi Tutupan Lahan",
     col = kelas_warna,
     legend = FALSE,
     labels = kelas_label)

#mengatur legenda pada plot
legend("right", 
       legend = kelas_label, 
       fill = kelas_warna, 
       title = "Kelas Tutupan Lahan", 
       cex = 0.5)

#margin pada graph neural network
plot(nn_model, main = "Arsitektur Neural Network") 


# export hasil klasifikasi tutupan lahan ke format GeoTIFF
writeRaster(predicted_raster, 
            filename = "hasil/Klasifikasi_output.tif", 
            overwrite = TRUE)

# Optional: Print confirmation message
cat("Klasifikasi citra telah diekspor ke Klasifikasi_Citra.tif\n")
