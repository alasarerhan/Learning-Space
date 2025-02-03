

#! AR(p) Autoregression
#! Bir zaman serisi kendinden önceki gecikmeler ile tahmin edilir.

#! Önceki zaman adımlarındaki gözlemlerin doğrusal bir kombinasyonu ile tahmin yapılır.
#! Trend ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.
#! p: zaman gecilmesi sayısıdır. p = 1 1 ise bir önceki zaman adımı ile model kurulmuş demek olur.

#! p = 1 -> yt = a1*yt-1 + et
#! p = 2 -> yt = a1*yt-1 + a2*yt-2 + et


#! MA(q) Moving Average
#! yt = m1*et-1 + et
#! yt = m1*et-1 + m2*et-2 + .... + mq*et-q + et
#! Önceki zaman adımlarında elde edilen hataların doğrusal bir kombinasyonu ile tahmin yapılır.
#! Trend ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.
#! q: Zaöam gecikmesi sayısıdır. 


#! ARMA(p,q) = AR(p) + MA(q)
#! yt = a1*yt-1 + m1*et-1 + et
#! ARMMA Simple Exponential Smoothing'e benzer. Fakat ARMA doğrusal bir regresyon formundadır.
#! Holt Winters yöntemlerinde terimler bir parametreye göre şekillenirken AR, MA, ARMA modellerinde terimlerin kendi katsayıları var.
#! Bu katsayıların bulunması ile ilgileniliyoruz. 
#! Aurotregressive Moving Average. AR ve MA yöntemlerini birleştirir.
#! Geçmiş değerler ve geçmiş hataların doğrusal bir kombinasyonu ile tahmin yapılır.
#! Trend ve mevsimsellik içermeyen tek değişkenli zaman serileri için uygundur.
#! p ve q zaman gecikmesi sayılarıdır. p AR modeli için, q MA modeli için gerekli olan gecikme sayılarıdır.


