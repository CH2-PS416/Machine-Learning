#Machine-Learning

##Data Understanding
  Untuk proses pengembangan model atau model development, dataset yang digunakan adalah data tourism Indonesia yang diambil dari kaggle yang dihimpun oleh tim GetLoc, salah satu tim Capstone Project Bangkit Academy 2021. Dataset ini berisi informasi mengenai tempat wisata yang dari beberapa kota di Indonesia yaitu Jakarta, Surabaya, Semarang, Bandung, dan Yogyakarta. Dataset ini juga terdiri dari 4 file yaitu :
1. pariwisata_ dengan _id.csv yang berisi informasi tempat wisata di 5 kota besar di Indonesia berjumlah ~400
2. user.csv yang berisi data pengguna dummy untuk membuat fitur rekomendasi berdasarkan pengguna
3. Tourism_rating.csv berisi 3 kolom yaitu user, place, dan rating yang diberikan, berfungsi untuk membuat sistem rekomendasi berdasarkan rating tersebut
4. package_tourism.csv berisi rekomendasi tempat terdekat berdasarkan waktu, biaya, dan rating.

##Preparation Data
  Tahap ini dilakukan Pengecekan missing value dan statistika deskriptif, selanjutya dilakukan penggabungan file menjadi kesatuan agar sesuai dengan pengembangan model yang dibuat. 
  
##Modeling 
###Model Development dengan Content Based Filtering
  Kami menggunakan count vetorize untuk mengonversi fitur penting setiap tempat wisata menjadi representasi vektor. Selanjutnya menghitung kemiripan antar item menggunakan metrik cosine similarity (derajat kesamaan) antar tempat wisata berdasarkan matriks count vectorize yang sudah dibuat sebelumnya.  Kemudian, dibuatkan dataframe dari hasil perhitungan cosine similarity dengan baris dan kolom berupa nama tempat wisata. Selanjutnya, dibuatkan fungsi  rekomendasi_wisata_by_keyword untuk menampilkan hasil rekomendasi berbasis konten berupa nama tempaat wisata dengan kategori yang diinginkan berdasarkan nama kota.
###Model Development dengan Collaborative Filtering
  Teknik ini membutuhkan data rating dari pengguna atau pembaca. Collaborative filtering adalah salah satu metode dalam sistem rekomendasi yang memprediksi preferensi atau minat pengguna terhadap item berdasarkan informasi dari pengguna lain (kolaborasi). Ide dasar dibalik collaborative filtering adalah bahwa pengguna yang memiliki preferensi serupa dalam masa lalu cenderung memiliki preferensi serupa untuk item di masa depan. Pada proyek ini akan dibuat model collaborative filtering berdasarkan kesamaan antar pengguna (User-Based Collaborative Filtering.
  Tahap pertama dilakukan pembagian data untuk data training dan data validasi, kemudian dilanjutkan dengan proses training model. Pada proses training, model menghitung skor kecocokan antara pengguna dan tempat wisata dengan teknik embedding. Pertama, dilakukan proses embedding terhadap data pengguna dan tempat wisata. Selanjutnya, melakukan operasi perkalian dot product antara embedding pengguna dan judul buku. Selain itu, ditambahkan bias untuk setiap pengguna dan judul buku. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Model dibuatkan class RecommenderNet dengan keras Model class. Model akan menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan Root Mean Squared Error (RMSE) sebagai metrik evaluasi.
  Berdasarkan hasil proses training model, didapat hasil yang cukup baik, diperoleh nilai Root Mean Squared Error (RMSE) sebesar sekitar 0.2939 dan RMSE pada data validasi sebesar 0.3353. Nilai ini cukup bagus untuk sistem rekomendasi. Untuk mengetahui hasil dari pengembangan model, langkah selanjutnya adalah mendapatkan rekomendasi judul buku berdasarkan model yang dikembangkan.
  Untuk mendapatkan rekomendasi tempat wisata, pertama diambil sampel user secara acak yang merupakan daftar tempat wisata yang belum pernah kunjungi oleh pengguna. Hal inilah yang akan menjadi tempat wisata yang direkomendasikan oleh sistem. Selanjutnya, untuk memperoleh rekomendasi tempat wisat, digunakan fungsi model.predict() dari library Keras dan hasil output akan menampilkan top-10 recommendation berdasarkan preferensi pengguna dan top-10 recommendation berdasarkan rating. 
##Evaluasi Model
###Content Based Filtering
  Metrik yang digunakan untuk evaluasi model dengan content based filtering di proyek kali ini adalah Precision, Recall, dan F1-Score. Metrik ini adalah metrik yang umum digunakan untuk mengukur kinerja model. Precision merupakan rasio item yang revelan yang dihasilkan oleh model terhadap total item yang dihasilkan. Recall merupakan rasio item relevan yang dihasilkan oleh model terhadap total item yang seharusnya direkomendasikan. Sedangkan, F1 Score adalah gabungan dari Precision dan Recall, memberikan nilai tunggal yang mengukur keseimbangan antara keduanya. Sebelum menghitung nilai evaluasi metrik menggunakan precision, recall dan f1 score, diperlukan sebuah data yang terdiri dari label sebenarnya dan digunakan untuk menilai hasil prediksi model, data ini disebut sebagai data ground truth. Data ground truth pada proyek ini dibuat menggunakan hasil derajat kesamaan yang dihitung menggunakan teknik cosine similarity, dimana setiap baris dan kolom mewakili judul buku, dan nilai di setiap sel pada dataframe mewakili label. Angka 1 untuk similar, dan angka 0 untuk tidak similar. Perlu ditetapkan juga sebuah nilai ambang batas atau threshold untuk memutuskan apakah nilai similarity antara dua item harus dianggap 1 (similar) atau 0 (tidak similar). Nilai ambang batas atau threshold ditetapkan sebesar 0.5 pada proyek ini. Nilai threshold ini disesuaikan dengan kebutuhan dan karakteristik setelah melihat hasil rekomendasi sebelumnya. Lalu dibuatkan matriks ground truth menggunakan fungsi np.where() dari NumPy. Matriks ini akan memiliki nilai 1 di posisi di mana nilai cosine similarity antara dua item lebih besar atau sama dengan nilai threshold yang ditetapkan, dan nilai 0 di posisi di mana nilai similarity di bawah threshold. Kemudian, setelah matriks dibuat, hasilnya disajikan dalam bentuk dataframe. Baris dan kolom Dataframe ground truth ini diindeks menggunakan judul buku dari data.

  Setelah dibuatkan matriks ground truth yang berisi label sebenarnya dari hasil cosine similarity. Selanjutnya, dilakukan proses perhitungan evaluasi model dengan metrik precision, recall, dan f1 score. Pertama, mengimport fungsi precision_recall_fscore_support dari library Sklearn yang digunakan untuk menghitung precision, recall dan f1 score. Lalu karena keterbatasan alokasi memori pada perangkat, data hanya diambil sekitar 10000 sampel dari cosine similarity dan ground truth matriks. Hal ini dilakukan untuk mempercepat proses perhitungan, terutama karena ukuran matriks yang cukup besar. Kemudian, matriks cosine similarity dan ground truth dikonversi menjadi array satu dimensi agar mempermudah perbandingan dan perhitungan metrik evaluasi.

Hasilnya disimpan dalam array predictions. Terakhir, digunakan fungsi precision_recall_fscore_support untuk menghitung precision, recall, dan f1 score. Parameter average='binary' digunakan karena sedang mengukur kinerja dalam konteks klasifikasi biner (1 atau 0). Parameter 'zero_division=1' digunakan untuk menghindari pembagian dengan nol jika ada kelas yang tidak terdapat di prediksi. Hasil evaluasi metriks didapat adalah sebagai berikut:

Precision: 1.0
Recall: 1.0
F1-score: 1.0
  Berdasarkan hasil evaluasi, didapat nilai dari masing - masing metrik evaluasi yaitu precision, recall dan F1 Score. Nilai Precision didapat sebesar 1.0, artinya semua prediksi positif model adalah benar dan tidak terdapat false positive. Nilai recall didapat nilai 1.0 menunjukkan bahwa model berhasil mengidentifikasi sekitar 100% dari semua item yang sebenarnya relevan. Nilai F1 Score didapat sekitar 1.0 juga, ini menunjukkan keseimbangan yang baik antara precision dan recall dan model cenderung memberikan hasil yang sangat baik untuk kedua kelas (positif dan negatif). Kesimpulannya, berdasarkan hasil metrik evaluasi tersebut model bekerja dengan sangat baik dalam memberikan rekomendasi item dengan content based filtering.
##Collaborative Filtering
  Seperti yang sudah dilihat pada proses pelatihan model di bagian modeling. Metrik yang digunakan untuk melakukan evaluasi model pada model dengan Collaborative Filtering di proyek ini adalah Root Mean Squared Error (RMSE). RMSE adalah metrik evaluasi yang umum digunakan untuk mengukur seberapa baik model memprediksi nilai kontinu dengan membandingkan nilai prediksi dengan nilai sebenarnya. Dalam konteks collaborative filtering, RMSE biasanya digunakan untuk mengukur seberapa baik model kolaboratif dalam memprediksi preferensi pengguna terhadap item.
Berdasarkan nilai RMSE yang didapat, Nilai ini cukup bagus untuk sistem rekomendasi sistem dan memiliki akurasi yang cukup baik. 
