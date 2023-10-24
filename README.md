# vocalemo

Speech Synthesis via [mimic3](https://pypi.org/project/mycroft-mimic3-tts/)

Install

```
virtualenv --python=python3 ~/.envs/.shift
source ~/.envs/.shift/bin/activate
cd vocalemo/
pip install -r requirements.txt
```

Demo

```
python demo.py
```

Output wav is saved as `synthesis_example.wav`

##

# Available voices - english

To generate this table run [generate_config.py](https://gitlab.audeering.com/project/shift/-/blob/_scratch/vocalemo/generate_config.py) to save `voices.json` & `assets/`
then [voices_table.py](https://gitlab.audeering.com/project/shift/-/blob/_scratch/vocalemo/voices_table.py) to save `wavs` and append table to `README.md`.

<table><tr><td>



</td><td>

 voice

</td><td>

 mimic3 TTS

</td><td>

 [Voice emotion](https://www.cs.columbia.edu/~hgs/audio/harvard.html)

</td><td>

 `arousal`

</td><td>

 `valence`

</td><td>

 `dominance`

</td><tr>
 <td>
 0
</td>
<td>

 ```
en_US/vctk_low#p330
```

</td><td>

<video src="wavs/en_US_vctk_p330.mov" width=120/>


https://github.com/audeering/shift/assets/93256324/bdc248b8-5a73-4d28-8bbe-0332ff7734cf




</td><td>

 <img src="./assets/en_US_vctk_p330.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.066
```

</td><td>

 ```
0.207
```

</td><td>

 ```
0.251
```

</td></tr>



<tr>
 <td>
 1
</td>
<td>

 ```
en_US/vctk_low#p267
```

</td><td>

 [wavs/en_US_vctk_p267.wav](wavs/en_US_vctk_p267.wav)

</td><td>

 <img src="./assets/en_US_vctk_p267.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.072
```

</td><td>

 ```
0.214
```

</td><td>

 ```
0.243
```

</td></tr>



<tr>
 <td>
 2
</td>
<td>

 ```
en_US/vctk_low#p230
```

</td><td>

 [wavs/en_US_vctk_p230.wav](wavs/en_US_vctk_p230.wav)

</td><td>

 <img src="./assets/en_US_vctk_p230.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.062
```

</td><td>

 ```
0.215
```

</td><td>

 ```
0.244
```

</td></tr>



<tr>
 <td>
 3
</td>
<td>

 ```
en_US/vctk_low#p351
```

</td><td>

 [wavs/en_US_vctk_p351.wav](wavs/en_US_vctk_p351.wav)

</td><td>

 <img src="./assets/en_US_vctk_p351.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.076
```

</td><td>

 ```
0.219
```

</td><td>

 ```
0.255
```

</td></tr>



<tr>
 <td>
 4
</td>
<td>

 ```
en_US/vctk_low#p306
```

</td><td>

 [wavs/en_US_vctk_p306.wav](wavs/en_US_vctk_p306.wav)

</td><td>

 <img src="./assets/en_US_vctk_p306.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.070
```

</td><td>

 ```
0.222
```

</td><td>

 ```
0.257
```

</td></tr>



<tr>
 <td>
 5
</td>
<td>

 ```
en_US/cmu-arctic_low#eey
```

</td><td>

 [wavs/en_US_cmu_arctic_eey.wav](wavs/en_US_cmu_arctic_eey.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_eey.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.075
```

</td><td>

 ```
0.222
```

</td><td>

 ```
0.304
```

</td></tr>



<tr>
 <td>
 6
</td>
<td>

 ```
en_US/vctk_low#p293
```

</td><td>

 [wavs/en_US_vctk_p293.wav](wavs/en_US_vctk_p293.wav)

</td><td>

 <img src="./assets/en_US_vctk_p293.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.072
```

</td><td>

 ```
0.222
```

</td><td>

 ```
0.261
```

</td></tr>



<tr>
 <td>
 7
</td>
<td>

 ```
en_US/vctk_low#p277
```

</td><td>

 [wavs/en_US_vctk_p277.wav](wavs/en_US_vctk_p277.wav)

</td><td>

 <img src="./assets/en_US_vctk_p277.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.074
```

</td><td>

 ```
0.223
```

</td><td>

 ```
0.261
```

</td></tr>



<tr>
 <td>
 8
</td>
<td>

 ```
en_US/vctk_low#p249
```

</td><td>

 [wavs/en_US_vctk_p249.wav](wavs/en_US_vctk_p249.wav)

</td><td>

 <img src="./assets/en_US_vctk_p249.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.063
```

</td><td>

 ```
0.229
```

</td><td>

 ```
0.266
```

</td></tr>



<tr>
 <td>
 9
</td>
<td>

 ```
en_US/vctk_low#p282
```

</td><td>

 [wavs/en_US_vctk_p282.wav](wavs/en_US_vctk_p282.wav)

</td><td>

 <img src="./assets/en_US_vctk_p282.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.077
```

</td><td>

 ```
0.230
```

</td><td>

 ```
0.253
```

</td></tr>



<tr>
 <td>
 10
</td>
<td>

 ```
en_US/cmu-arctic_low#clb
```

</td><td>

 [wavs/en_US_cmu_arctic_clb.wav](wavs/en_US_cmu_arctic_clb.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_clb.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.072
```

</td><td>

 ```
0.230
```

</td><td>

 ```
0.285
```

</td></tr>



<tr>
 <td>
 11
</td>
<td>

 ```
en_US/vctk_low#p343
```

</td><td>

 [wavs/en_US_vctk_p343.wav](wavs/en_US_vctk_p343.wav)

</td><td>

 <img src="./assets/en_US_vctk_p343.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.069
```

</td><td>

 ```
0.231
```

</td><td>

 ```
0.260
```

</td></tr>



<tr>
 <td>
 12
</td>
<td>

 ```
en_US/vctk_low#p264
```

</td><td>

 [wavs/en_US_vctk_p264.wav](wavs/en_US_vctk_p264.wav)

</td><td>

 <img src="./assets/en_US_vctk_p264.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.082
```

</td><td>

 ```
0.231
```

</td><td>

 ```
0.258
```

</td></tr>



<tr>
 <td>
 13
</td>
<td>

 ```
en_US/vctk_low#p313
```

</td><td>

 [wavs/en_US_vctk_p313.wav](wavs/en_US_vctk_p313.wav)

</td><td>

 <img src="./assets/en_US_vctk_p313.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.067
```

</td><td>

 ```
0.233
```

</td><td>

 ```
0.256
```

</td></tr>



<tr>
 <td>
 14
</td>
<td>

 ```
en_US/cmu-arctic_low#lnh
```

</td><td>

 [wavs/en_US_cmu_arctic_lnh.wav](wavs/en_US_cmu_arctic_lnh.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_lnh.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.104
```

</td><td>

 ```
0.233
```

</td><td>

 ```
0.269
```

</td></tr>



<tr>
 <td>
 15
</td>
<td>

 ```
en_US/vctk_low#p273
```

</td><td>

 [wavs/en_US_vctk_p273.wav](wavs/en_US_vctk_p273.wav)

</td><td>

 <img src="./assets/en_US_vctk_p273.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.066
```

</td><td>

 ```
0.234
```

</td><td>

 ```
0.258
```

</td></tr>



<tr>
 <td>
 16
</td>
<td>

 ```
en_US/vctk_low#p329
```

</td><td>

 [wavs/en_US_vctk_p329.wav](wavs/en_US_vctk_p329.wav)

</td><td>

 <img src="./assets/en_US_vctk_p329.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.072
```

</td><td>

 ```
0.234
```

</td><td>

 ```
0.265
```

</td></tr>



<tr>
 <td>
 17
</td>
<td>

 ```
en_US/m-ailabs_low#judy_bieber
```

</td><td>

 [wavs/en_US_m-ailabs_judy_bieber.wav](wavs/en_US_m-ailabs_judy_bieber.wav)

</td><td>

 <img src="./assets/en_US_m-ailabs_judy_bieber.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.127
```

</td><td>

 ```
0.235
```

</td><td>

 ```
0.307
```

</td></tr>



<tr>
 <td>
 18
</td>
<td>

 ```
en_US/vctk_low#p228
```

</td><td>

 [wavs/en_US_vctk_p228.wav](wavs/en_US_vctk_p228.wav)

</td><td>

 <img src="./assets/en_US_vctk_p228.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.075
```

</td><td>

 ```
0.238
```

</td><td>

 ```
0.252
```

</td></tr>



<tr>
 <td>
 19
</td>
<td>

 ```
en_US/vctk_low#p234
```

</td><td>

 [wavs/en_US_vctk_p234.wav](wavs/en_US_vctk_p234.wav)

</td><td>

 <img src="./assets/en_US_vctk_p234.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.080
```

</td><td>

 ```
0.238
```

</td><td>

 ```
0.237
```

</td></tr>



<tr>
 <td>
 20
</td>
<td>

 ```
en_US/vctk_low#p283
```

</td><td>

 [wavs/en_US_vctk_p283.wav](wavs/en_US_vctk_p283.wav)

</td><td>

 <img src="./assets/en_US_vctk_p283.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.075
```

</td><td>

 ```
0.239
```

</td><td>

 ```
0.269
```

</td></tr>



<tr>
 <td>
 21
</td>
<td>

 ```
en_US/vctk_low#p297
```

</td><td>

 [wavs/en_US_vctk_p297.wav](wavs/en_US_vctk_p297.wav)

</td><td>

 <img src="./assets/en_US_vctk_p297.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.079
```

</td><td>

 ```
0.239
```

</td><td>

 ```
0.275
```

</td></tr>



<tr>
 <td>
 22
</td>
<td>

 ```
en_US/cmu-arctic_low#ljm
```

</td><td>

 [wavs/en_US_cmu_arctic_ljm.wav](wavs/en_US_cmu_arctic_ljm.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_ljm.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.098
```

</td><td>

 ```
0.239
```

</td><td>

 ```
0.283
```

</td></tr>



<tr>
 <td>
 23
</td>
<td>

 ```
en_US/vctk_low#p270
```

</td><td>

 [wavs/en_US_vctk_p270.wav](wavs/en_US_vctk_p270.wav)

</td><td>

 <img src="./assets/en_US_vctk_p270.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.066
```

</td><td>

 ```
0.240
```

</td><td>

 ```
0.261
```

</td></tr>



<tr>
 <td>
 24
</td>
<td>

 ```
en_US/vctk_low#p364
```

</td><td>

 [wavs/en_US_vctk_p364.wav](wavs/en_US_vctk_p364.wav)

</td><td>

 <img src="./assets/en_US_vctk_p364.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.058
```

</td><td>

 ```
0.241
```

</td><td>

 ```
0.285
```

</td></tr>



<tr>
 <td>
 25
</td>
<td>

 ```
en_US/vctk_low#p295
```

</td><td>

 [wavs/en_US_vctk_p295.wav](wavs/en_US_vctk_p295.wav)

</td><td>

 <img src="./assets/en_US_vctk_p295.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.073
```

</td><td>

 ```
0.242
```

</td><td>

 ```
0.244
```

</td></tr>



<tr>
 <td>
 26
</td>
<td>

 ```
en_US/vctk_low#p341
```

</td><td>

 [wavs/en_US_vctk_p341.wav](wavs/en_US_vctk_p341.wav)

</td><td>

 <img src="./assets/en_US_vctk_p341.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.079
```

</td><td>

 ```
0.242
```

</td><td>

 ```
0.246
```

</td></tr>



<tr>
 <td>
 27
</td>
<td>

 ```
en_US/vctk_low#p362
```

</td><td>

 [wavs/en_US_vctk_p362.wav](wavs/en_US_vctk_p362.wav)

</td><td>

 <img src="./assets/en_US_vctk_p362.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.087
```

</td><td>

 ```
0.242
```

</td><td>

 ```
0.248
```

</td></tr>



<tr>
 <td>
 28
</td>
<td>

 ```
en_US/vctk_low#p314
```

</td><td>

 [wavs/en_US_vctk_p314.wav](wavs/en_US_vctk_p314.wav)

</td><td>

 <img src="./assets/en_US_vctk_p314.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.081
```

</td><td>

 ```
0.243
```

</td><td>

 ```
0.240
```

</td></tr>



<tr>
 <td>
 29
</td>
<td>

 ```
en_US/vctk_low#p245
```

</td><td>

 [wavs/en_US_vctk_p245.wav](wavs/en_US_vctk_p245.wav)

</td><td>

 <img src="./assets/en_US_vctk_p245.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.061
```

</td><td>

 ```
0.244
```

</td><td>

 ```
0.293
```

</td></tr>



<tr>
 <td>
 30
</td>
<td>

 ```
en_US/vctk_low#p269
```

</td><td>

 [wavs/en_US_vctk_p269.wav](wavs/en_US_vctk_p269.wav)

</td><td>

 <img src="./assets/en_US_vctk_p269.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.083
```

</td><td>

 ```
0.246
```

</td><td>

 ```
0.250
```

</td></tr>



<tr>
 <td>
 31
</td>
<td>

 ```
en_US/vctk_low#p308
```

</td><td>

 [wavs/en_US_vctk_p308.wav](wavs/en_US_vctk_p308.wav)

</td><td>

 <img src="./assets/en_US_vctk_p308.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.077
```

</td><td>

 ```
0.246
```

</td><td>

 ```
0.268
```

</td></tr>



<tr>
 <td>
 32
</td>
<td>

 ```
en_US/vctk_low#p301
```

</td><td>

 [wavs/en_US_vctk_p301.wav](wavs/en_US_vctk_p301.wav)

</td><td>

 <img src="./assets/en_US_vctk_p301.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.079
```

</td><td>

 ```
0.247
```

</td><td>

 ```
0.261
```

</td></tr>



<tr>
 <td>
 33
</td>
<td>

 ```
en_US/vctk_low#p262
```

</td><td>

 [wavs/en_US_vctk_p262.wav](wavs/en_US_vctk_p262.wav)

</td><td>

 <img src="./assets/en_US_vctk_p262.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.072
```

</td><td>

 ```
0.249
```

</td><td>

 ```
0.260
```

</td></tr>



<tr>
 <td>
 34
</td>
<td>

 ```
en_US/vctk_low#p271
```

</td><td>

 [wavs/en_US_vctk_p271.wav](wavs/en_US_vctk_p271.wav)

</td><td>

 <img src="./assets/en_US_vctk_p271.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.065
```

</td><td>

 ```
0.250
```

</td><td>

 ```
0.256
```

</td></tr>



<tr>
 <td>
 35
</td>
<td>

 ```
en_US/vctk_low#p268
```

</td><td>

 [wavs/en_US_vctk_p268.wav](wavs/en_US_vctk_p268.wav)

</td><td>

 <img src="./assets/en_US_vctk_p268.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.099
```

</td><td>

 ```
0.250
```

</td><td>

 ```
0.261
```

</td></tr>



<tr>
 <td>
 36
</td>
<td>

 ```
en_US/vctk_low#p303
```

</td><td>

 [wavs/en_US_vctk_p303.wav](wavs/en_US_vctk_p303.wav)

</td><td>

 <img src="./assets/en_US_vctk_p303.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.076
```

</td><td>

 ```
0.250
```

</td><td>

 ```
0.263
```

</td></tr>



<tr>
 <td>
 37
</td>
<td>

 ```
en_US/cmu-arctic_low#awbrms
```

</td><td>

 [wavs/en_US_cmu_arctic_awbrms.wav](wavs/en_US_cmu_arctic_awbrms.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_awbrms.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.087
```

</td><td>

 ```
0.251
```

</td><td>

 ```
0.291
```

</td></tr>



<tr>
 <td>
 38
</td>
<td>

 ```
en_US/vctk_low#p266
```

</td><td>

 [wavs/en_US_vctk_p266.wav](wavs/en_US_vctk_p266.wav)

</td><td>

 <img src="./assets/en_US_vctk_p266.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.070
```

</td><td>

 ```
0.251
```

</td><td>

 ```
0.277
```

</td></tr>



<tr>
 <td>
 39
</td>
<td>

 ```
en_US/vctk_low#p280
```

</td><td>

 [wavs/en_US_vctk_p280.wav](wavs/en_US_vctk_p280.wav)

</td><td>

 <img src="./assets/en_US_vctk_p280.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.080
```

</td><td>

 ```
0.251
```

</td><td>

 ```
0.250
```

</td></tr>



<tr>
 <td>
 40
</td>
<td>

 ```
en_US/vctk_low#p276
```

</td><td>

 [wavs/en_US_vctk_p276.wav](wavs/en_US_vctk_p276.wav)

</td><td>

 <img src="./assets/en_US_vctk_p276.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.087
```

</td><td>

 ```
0.252
```

</td><td>

 ```
0.279
```

</td></tr>



<tr>
 <td>
 41
</td>
<td>

 ```
en_US/vctk_low#p238
```

</td><td>

 [wavs/en_US_vctk_p238.wav](wavs/en_US_vctk_p238.wav)

</td><td>

 <img src="./assets/en_US_vctk_p238.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.099
```

</td><td>

 ```
0.252
```

</td><td>

 ```
0.271
```

</td></tr>



<tr>
 <td>
 42
</td>
<td>

 ```
en_US/vctk_low#p257
```

</td><td>

 [wavs/en_US_vctk_p257.wav](wavs/en_US_vctk_p257.wav)

</td><td>

 <img src="./assets/en_US_vctk_p257.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.087
```

</td><td>

 ```
0.252
```

</td><td>

 ```
0.259
```

</td></tr>



<tr>
 <td>
 43
</td>
<td>

 ```
en_US/vctk_low#p256
```

</td><td>

 [wavs/en_US_vctk_p256.wav](wavs/en_US_vctk_p256.wav)

</td><td>

 <img src="./assets/en_US_vctk_p256.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.066
```

</td><td>

 ```
0.252
```

</td><td>

 ```
0.278
```

</td></tr>



<tr>
 <td>
 44
</td>
<td>

 ```
en_US/cmu-arctic_low#rxr
```

</td><td>

 [wavs/en_US_cmu_arctic_rxr.wav](wavs/en_US_cmu_arctic_rxr.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_rxr.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.080
```

</td><td>

 ```
0.252
```

</td><td>

 ```
0.267
```

</td></tr>



<tr>
 <td>
 45
</td>
<td>

 ```
en_US/cmu-arctic_low#gka
```

</td><td>

 [wavs/en_US_cmu_arctic_gka.wav](wavs/en_US_cmu_arctic_gka.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_gka.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.076
```

</td><td>

 ```
0.253
```

</td><td>

 ```
0.292
```

</td></tr>



<tr>
 <td>
 46
</td>
<td>

 ```
en_US/vctk_low#p229
```

</td><td>

 [wavs/en_US_vctk_p229.wav](wavs/en_US_vctk_p229.wav)

</td><td>

 <img src="./assets/en_US_vctk_p229.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.067
```

</td><td>

 ```
0.253
```

</td><td>

 ```
0.291
```

</td></tr>



<tr>
 <td>
 47
</td>
<td>

 ```
en_US/vctk_low#p317
```

</td><td>

 [wavs/en_US_vctk_p317.wav](wavs/en_US_vctk_p317.wav)

</td><td>

 <img src="./assets/en_US_vctk_p317.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.095
```

</td><td>

 ```
0.253
```

</td><td>

 ```
0.229
```

</td></tr>



<tr>
 <td>
 48
</td>
<td>

 ```
en_US/vctk_low#p231
```

</td><td>

 [wavs/en_US_vctk_p231.wav](wavs/en_US_vctk_p231.wav)

</td><td>

 <img src="./assets/en_US_vctk_p231.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.082
```

</td><td>

 ```
0.253
```

</td><td>

 ```
0.253
```

</td></tr>



<tr>
 <td>
 49
</td>
<td>

 ```
en_US/cmu-arctic_low#jmk
```

</td><td>

 [wavs/en_US_cmu_arctic_jmk.wav](wavs/en_US_cmu_arctic_jmk.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_jmk.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.070
```

</td><td>

 ```
0.254
```

</td><td>

 ```
0.329
```

</td></tr>



<tr>
 <td>
 50
</td>
<td>

 ```
en_US/vctk_low#p347
```

</td><td>

 [wavs/en_US_vctk_p347.wav](wavs/en_US_vctk_p347.wav)

</td><td>

 <img src="./assets/en_US_vctk_p347.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.083
```

</td><td>

 ```
0.254
```

</td><td>

 ```
0.274
```

</td></tr>



<tr>
 <td>
 51
</td>
<td>

 ```
en_US/cmu-arctic_low#ksp
```

</td><td>

 [wavs/en_US_cmu_arctic_ksp.wav](wavs/en_US_cmu_arctic_ksp.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_ksp.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.080
```

</td><td>

 ```
0.255
```

</td><td>

 ```
0.297
```

</td></tr>



<tr>
 <td>
 52
</td>
<td>

 ```
en_US/vctk_low#p225
```

</td><td>

 [wavs/en_US_vctk_p225.wav](wavs/en_US_vctk_p225.wav)

</td><td>

 <img src="./assets/en_US_vctk_p225.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.091
```

</td><td>

 ```
0.255
```

</td><td>

 ```
0.258
```

</td></tr>



<tr>
 <td>
 53
</td>
<td>

 ```
en_US/vctk_low#p227
```

</td><td>

 [wavs/en_US_vctk_p227.wav](wavs/en_US_vctk_p227.wav)

</td><td>

 <img src="./assets/en_US_vctk_p227.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.070
```

</td><td>

 ```
0.256
```

</td><td>

 ```
0.258
```

</td></tr>



<tr>
 <td>
 54
</td>
<td>

 ```
en_US/vctk_low#p294
```

</td><td>

 [wavs/en_US_vctk_p294.wav](wavs/en_US_vctk_p294.wav)

</td><td>

 <img src="./assets/en_US_vctk_p294.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.084
```

</td><td>

 ```
0.256
```

</td><td>

 ```
0.253
```

</td></tr>



<tr>
 <td>
 55
</td>
<td>

 ```
en_US/vctk_low#p275
```

</td><td>

 [wavs/en_US_vctk_p275.wav](wavs/en_US_vctk_p275.wav)

</td><td>

 <img src="./assets/en_US_vctk_p275.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.066
```

</td><td>

 ```
0.257
```

</td><td>

 ```
0.270
```

</td></tr>



<tr>
 <td>
 56
</td>
<td>

 ```
en_US/cmu-arctic_low#aew
```

</td><td>

 [wavs/en_US_cmu_arctic_aew.wav](wavs/en_US_cmu_arctic_aew.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_aew.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.072
```

</td><td>

 ```
0.257
```

</td><td>

 ```
0.301
```

</td></tr>



<tr>
 <td>
 57
</td>
<td>

 ```
en_US/vctk_low#p255
```

</td><td>

 [wavs/en_US_vctk_p255.wav](wavs/en_US_vctk_p255.wav)

</td><td>

 <img src="./assets/en_US_vctk_p255.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.079
```

</td><td>

 ```
0.258
```

</td><td>

 ```
0.267
```

</td></tr>



<tr>
 <td>
 58
</td>
<td>

 ```
en_US/vctk_low#p374
```

</td><td>

 [wavs/en_US_vctk_p374.wav](wavs/en_US_vctk_p374.wav)

</td><td>

 <img src="./assets/en_US_vctk_p374.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.071
```

</td><td>

 ```
0.258
```

</td><td>

 ```
0.261
```

</td></tr>



<tr>
 <td>
 59
</td>
<td>

 ```
en_US/cmu-arctic_low#slt
```

</td><td>

 [wavs/en_US_cmu_arctic_slt.wav](wavs/en_US_cmu_arctic_slt.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_slt.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.081
```

</td><td>

 ```
0.259
```

</td><td>

 ```
0.306
```

</td></tr>



<tr>
 <td>
 60
</td>
<td>

 ```
en_US/vctk_low#p237
```

</td><td>

 [wavs/en_US_vctk_p237.wav](wavs/en_US_vctk_p237.wav)

</td><td>

 <img src="./assets/en_US_vctk_p237.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.080
```

</td><td>

 ```
0.259
```

</td><td>

 ```
0.287
```

</td></tr>



<tr>
 <td>
 61
</td>
<td>

 ```
en_US/vctk_low#p259
```

</td><td>

 [wavs/en_US_vctk_p259.wav](wavs/en_US_vctk_p259.wav)

</td><td>

 <img src="./assets/en_US_vctk_p259.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.065
```

</td><td>

 ```
0.260
```

</td><td>

 ```
0.263
```

</td></tr>



<tr>
 <td>
 62
</td>
<td>

 ```
en_US/vctk_low#p281
```

</td><td>

 [wavs/en_US_vctk_p281.wav](wavs/en_US_vctk_p281.wav)

</td><td>

 <img src="./assets/en_US_vctk_p281.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.070
```

</td><td>

 ```
0.260
```

</td><td>

 ```
0.258
```

</td></tr>



<tr>
 <td>
 63
</td>
<td>

 ```
en_US/vctk_low#p335
```

</td><td>

 [wavs/en_US_vctk_p335.wav](wavs/en_US_vctk_p335.wav)

</td><td>

 <img src="./assets/en_US_vctk_p335.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.097
```

</td><td>

 ```
0.261
```

</td><td>

 ```
0.288
```

</td></tr>



<tr>
 <td>
 64
</td>
<td>

 ```
en_US/m-ailabs_low#mary_ann
```

</td><td>

 [wavs/en_US_m-ailabs_mary_ann.wav](wavs/en_US_m-ailabs_mary_ann.wav)

</td><td>

 <img src="./assets/en_US_m-ailabs_mary_ann.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.093
```

</td><td>

 ```
0.262
```

</td><td>

 ```
0.320
```

</td></tr>



<tr>
 <td>
 65
</td>
<td>

 ```
en_US/vctk_low#p318
```

</td><td>

 [wavs/en_US_vctk_p318.wav](wavs/en_US_vctk_p318.wav)

</td><td>

 <img src="./assets/en_US_vctk_p318.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.098
```

</td><td>

 ```
0.263
```

</td><td>

 ```
0.243
```

</td></tr>



<tr>
 <td>
 66
</td>
<td>

 ```
en_US/vctk_low#p252
```

</td><td>

 [wavs/en_US_vctk_p252.wav](wavs/en_US_vctk_p252.wav)

</td><td>

 <img src="./assets/en_US_vctk_p252.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.073
```

</td><td>

 ```
0.263
```

</td><td>

 ```
0.243
```

</td></tr>



<tr>
 <td>
 67
</td>
<td>

 ```
en_US/vctk_low#p326
```

</td><td>

 [wavs/en_US_vctk_p326.wav](wavs/en_US_vctk_p326.wav)

</td><td>

 <img src="./assets/en_US_vctk_p326.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.080
```

</td><td>

 ```
0.264
```

</td><td>

 ```
0.272
```

</td></tr>



<tr>
 <td>
 68
</td>
<td>

 ```
en_US/vctk_low#p304
```

</td><td>

 [wavs/en_US_vctk_p304.wav](wavs/en_US_vctk_p304.wav)

</td><td>

 <img src="./assets/en_US_vctk_p304.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.068
```

</td><td>

 ```
0.265
```

</td><td>

 ```
0.279
```

</td></tr>



<tr>
 <td>
 69
</td>
<td>

 ```
en_US/vctk_low#p274
```

</td><td>

 [wavs/en_US_vctk_p274.wav](wavs/en_US_vctk_p274.wav)

</td><td>

 <img src="./assets/en_US_vctk_p274.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.069
```

</td><td>

 ```
0.265
```

</td><td>

 ```
0.269
```

</td></tr>



<tr>
 <td>
 70
</td>
<td>

 ```
en_US/vctk_low#p339
```

</td><td>

 [wavs/en_US_vctk_p339.wav](wavs/en_US_vctk_p339.wav)

</td><td>

 <img src="./assets/en_US_vctk_p339.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.101
```

</td><td>

 ```
0.265
```

</td><td>

 ```
0.261
```

</td></tr>



<tr>
 <td>
 71
</td>
<td>

 ```
en_US/vctk_low#p334
```

</td><td>

 [wavs/en_US_vctk_p334.wav](wavs/en_US_vctk_p334.wav)

</td><td>

 <img src="./assets/en_US_vctk_p334.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.077
```

</td><td>

 ```
0.266
```

</td><td>

 ```
0.255
```

</td></tr>



<tr>
 <td>
 72
</td>
<td>

 ```
en_US/vctk_low#p263
```

</td><td>

 [wavs/en_US_vctk_p263.wav](wavs/en_US_vctk_p263.wav)

</td><td>

 <img src="./assets/en_US_vctk_p263.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.069
```

</td><td>

 ```
0.268
```

</td><td>

 ```
0.293
```

</td></tr>



<tr>
 <td>
 73
</td>
<td>

 ```
en_US/vctk_low#p239
```

</td><td>

 [wavs/en_US_vctk_p239.wav](wavs/en_US_vctk_p239.wav)

</td><td>

 <img src="./assets/en_US_vctk_p239.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.092
```

</td><td>

 ```
0.268
```

</td><td>

 ```
0.282
```

</td></tr>



<tr>
 <td>
 74
</td>
<td>

 ```
en_US/vctk_low#p265
```

</td><td>

 [wavs/en_US_vctk_p265.wav](wavs/en_US_vctk_p265.wav)

</td><td>

 <img src="./assets/en_US_vctk_p265.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.093
```

</td><td>

 ```
0.269
```

</td><td>

 ```
0.279
```

</td></tr>



<tr>
 <td>
 75
</td>
<td>

 ```
en_US/vctk_low#p251
```

</td><td>

 [wavs/en_US_vctk_p251.wav](wavs/en_US_vctk_p251.wav)

</td><td>

 <img src="./assets/en_US_vctk_p251.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.078
```

</td><td>

 ```
0.269
```

</td><td>

 ```
0.255
```

</td></tr>



<tr>
 <td>
 76
</td>
<td>

 ```
en_US/cmu-arctic_low#ahw
```

</td><td>

 [wavs/en_US_cmu_arctic_ahw.wav](wavs/en_US_cmu_arctic_ahw.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_ahw.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.092
```

</td><td>

 ```
0.269
```

</td><td>

 ```
0.297
```

</td></tr>



<tr>
 <td>
 77
</td>
<td>

 ```
en_US/vctk_low#p233
```

</td><td>

 [wavs/en_US_vctk_p233.wav](wavs/en_US_vctk_p233.wav)

</td><td>

 <img src="./assets/en_US_vctk_p233.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.094
```

</td><td>

 ```
0.270
```

</td><td>

 ```
0.262
```

</td></tr>



<tr>
 <td>
 78
</td>
<td>

 ```
en_US/vctk_low#p284
```

</td><td>

 [wavs/en_US_vctk_p284.wav](wavs/en_US_vctk_p284.wav)

</td><td>

 <img src="./assets/en_US_vctk_p284.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.070
```

</td><td>

 ```
0.270
```

</td><td>

 ```
0.273
```

</td></tr>



<tr>
 <td>
 79
</td>
<td>

 ```
en_US/cmu-arctic_low#fem
```

</td><td>

 [wavs/en_US_cmu_arctic_fem.wav](wavs/en_US_cmu_arctic_fem.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_fem.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.075
```

</td><td>

 ```
0.271
```

</td><td>

 ```
0.292
```

</td></tr>



<tr>
 <td>
 80
</td>
<td>

 ```
en_US/vctk_low#p340
```

</td><td>

 [wavs/en_US_vctk_p340.wav](wavs/en_US_vctk_p340.wav)

</td><td>

 <img src="./assets/en_US_vctk_p340.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.091
```

</td><td>

 ```
0.271
```

</td><td>

 ```
0.263
```

</td></tr>



<tr>
 <td>
 81
</td>
<td>

 ```
en_US/vctk_low#p278
```

</td><td>

 [wavs/en_US_vctk_p278.wav](wavs/en_US_vctk_p278.wav)

</td><td>

 <img src="./assets/en_US_vctk_p278.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.067
```

</td><td>

 ```
0.271
```

</td><td>

 ```
0.273
```

</td></tr>



<tr>
 <td>
 82
</td>
<td>

 ```
en_US/vctk_low#p272
```

</td><td>

 [wavs/en_US_vctk_p272.wav](wavs/en_US_vctk_p272.wav)

</td><td>

 <img src="./assets/en_US_vctk_p272.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.068
```

</td><td>

 ```
0.271
```

</td><td>

 ```
0.288
```

</td></tr>



<tr>
 <td>
 83
</td>
<td>

 ```
en_US/vctk_low#p226
```

</td><td>

 [wavs/en_US_vctk_p226.wav](wavs/en_US_vctk_p226.wav)

</td><td>

 <img src="./assets/en_US_vctk_p226.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.077
```

</td><td>

 ```
0.272
```

</td><td>

 ```
0.262
```

</td></tr>



<tr>
 <td>
 84
</td>
<td>

 ```
en_US/vctk_low#p376
```

</td><td>

 [wavs/en_US_vctk_p376.wav](wavs/en_US_vctk_p376.wav)

</td><td>

 <img src="./assets/en_US_vctk_p376.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.071
```

</td><td>

 ```
0.272
```

</td><td>

 ```
0.273
```

</td></tr>



<tr>
 <td>
 85
</td>
<td>

 ```
en_US/vctk_low#p305
```

</td><td>

 [wavs/en_US_vctk_p305.wav](wavs/en_US_vctk_p305.wav)

</td><td>

 <img src="./assets/en_US_vctk_p305.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.103
```

</td><td>

 ```
0.273
```

</td><td>

 ```
0.261
```

</td></tr>



<tr>
 <td>
 86
</td>
<td>

 ```
en_US/vctk_low#p241
```

</td><td>

 [wavs/en_US_vctk_p241.wav](wavs/en_US_vctk_p241.wav)

</td><td>

 <img src="./assets/en_US_vctk_p241.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.072
```

</td><td>

 ```
0.273
```

</td><td>

 ```
0.262
```

</td></tr>



<tr>
 <td>
 87
</td>
<td>

 ```
en_US/vctk_low#p240
```

</td><td>

 [wavs/en_US_vctk_p240.wav](wavs/en_US_vctk_p240.wav)

</td><td>

 <img src="./assets/en_US_vctk_p240.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.097
```

</td><td>

 ```
0.274
```

</td><td>

 ```
0.268
```

</td></tr>



<tr>
 <td>
 88
</td>
<td>

 ```
en_US/vctk_low#p311
```

</td><td>

 [wavs/en_US_vctk_p311.wav](wavs/en_US_vctk_p311.wav)

</td><td>

 <img src="./assets/en_US_vctk_p311.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.074
```

</td><td>

 ```
0.274
```

</td><td>

 ```
0.253
```

</td></tr>



<tr>
 <td>
 89
</td>
<td>

 ```
en_US/vctk_low#p336
```

</td><td>

 [wavs/en_US_vctk_p336.wav](wavs/en_US_vctk_p336.wav)

</td><td>

 <img src="./assets/en_US_vctk_p336.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.090
```

</td><td>

 ```
0.274
```

</td><td>

 ```
0.273
```

</td></tr>



<tr>
 <td>
 90
</td>
<td>

 ```
en_US/vctk_low#p258
```

</td><td>

 [wavs/en_US_vctk_p258.wav](wavs/en_US_vctk_p258.wav)

</td><td>

 <img src="./assets/en_US_vctk_p258.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.078
```

</td><td>

 ```
0.274
```

</td><td>

 ```
0.265
```

</td></tr>



<tr>
 <td>
 91
</td>
<td>

 ```
en_US/vctk_low#p312
```

</td><td>

 [wavs/en_US_vctk_p312.wav](wavs/en_US_vctk_p312.wav)

</td><td>

 <img src="./assets/en_US_vctk_p312.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.109
```

</td><td>

 ```
0.274
```

</td><td>

 ```
0.237
```

</td></tr>



<tr>
 <td>
 92
</td>
<td>

 ```
en_US/vctk_low#p310
```

</td><td>

 [wavs/en_US_vctk_p310.wav](wavs/en_US_vctk_p310.wav)

</td><td>

 <img src="./assets/en_US_vctk_p310.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.108
```

</td><td>

 ```
0.276
```

</td><td>

 ```
0.260
```

</td></tr>



<tr>
 <td>
 93
</td>
<td>

 ```
en_US/vctk_low#p298
```

</td><td>

 [wavs/en_US_vctk_p298.wav](wavs/en_US_vctk_p298.wav)

</td><td>

 <img src="./assets/en_US_vctk_p298.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.074
```

</td><td>

 ```
0.276
```

</td><td>

 ```
0.274
```

</td></tr>



<tr>
 <td>
 94
</td>
<td>

 ```
en_US/vctk_low#p246
```

</td><td>

 [wavs/en_US_vctk_p246.wav](wavs/en_US_vctk_p246.wav)

</td><td>

 <img src="./assets/en_US_vctk_p246.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.074
```

</td><td>

 ```
0.276
```

</td><td>

 ```
0.252
```

</td></tr>



<tr>
 <td>
 95
</td>
<td>

 ```
en_US/vctk_low#p261
```

</td><td>

 [wavs/en_US_vctk_p261.wav](wavs/en_US_vctk_p261.wav)

</td><td>

 <img src="./assets/en_US_vctk_p261.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.121
```

</td><td>

 ```
0.277
```

</td><td>

 ```
0.271
```

</td></tr>



<tr>
 <td>
 96
</td>
<td>

 ```
en_US/vctk_low#p253
```

</td><td>

 [wavs/en_US_vctk_p253.wav](wavs/en_US_vctk_p253.wav)

</td><td>

 <img src="./assets/en_US_vctk_p253.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.106
```

</td><td>

 ```
0.277
```

</td><td>

 ```
0.250
```

</td></tr>



<tr>
 <td>
 97
</td>
<td>

 ```
en_US/vctk_low#p316
```

</td><td>

 [wavs/en_US_vctk_p316.wav](wavs/en_US_vctk_p316.wav)

</td><td>

 <img src="./assets/en_US_vctk_p316.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.070
```

</td><td>

 ```
0.278
```

</td><td>

 ```
0.283
```

</td></tr>



<tr>
 <td>
 98
</td>
<td>

 ```
en_US/vctk_low#p360
```

</td><td>

 [wavs/en_US_vctk_p360.wav](wavs/en_US_vctk_p360.wav)

</td><td>

 <img src="./assets/en_US_vctk_p360.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.077
```

</td><td>

 ```
0.278
```

</td><td>

 ```
0.268
```

</td></tr>



<tr>
 <td>
 99
</td>
<td>

 ```
en_US/vctk_low#p307
```

</td><td>

 [wavs/en_US_vctk_p307.wav](wavs/en_US_vctk_p307.wav)

</td><td>

 <img src="./assets/en_US_vctk_p307.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.111
```

</td><td>

 ```
0.279
```

</td><td>

 ```
0.253
```

</td></tr>



<tr>
 <td>
 100
</td>
<td>

 ```
en_US/cmu-arctic_low#bdl
```

</td><td>

 [wavs/en_US_cmu_arctic_bdl.wav](wavs/en_US_cmu_arctic_bdl.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_bdl.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.079
```

</td><td>

 ```
0.279
```

</td><td>

 ```
0.296
```

</td></tr>



<tr>
 <td>
 101
</td>
<td>

 ```
en_US/vctk_low#p292
```

</td><td>

 [wavs/en_US_vctk_p292.wav](wavs/en_US_vctk_p292.wav)

</td><td>

 <img src="./assets/en_US_vctk_p292.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.071
```

</td><td>

 ```
0.281
```

</td><td>

 ```
0.276
```

</td></tr>



<tr>
 <td>
 102
</td>
<td>

 ```
en_US/vctk_low#p323
```

</td><td>

 [wavs/en_US_vctk_p323.wav](wavs/en_US_vctk_p323.wav)

</td><td>

 <img src="./assets/en_US_vctk_p323.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.131
```

</td><td>

 ```
0.281
```

</td><td>

 ```
0.263
```

</td></tr>



<tr>
 <td>
 103
</td>
<td>

 ```
en_US/vctk_low#p285
```

</td><td>

 [wavs/en_US_vctk_p285.wav](wavs/en_US_vctk_p285.wav)

</td><td>

 <img src="./assets/en_US_vctk_p285.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.072
```

</td><td>

 ```
0.281
```

</td><td>

 ```
0.271
```

</td></tr>



<tr>
 <td>
 104
</td>
<td>

 ```
en_US/vctk_low#p243
```

</td><td>

 [wavs/en_US_vctk_p243.wav](wavs/en_US_vctk_p243.wav)

</td><td>

 <img src="./assets/en_US_vctk_p243.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.080
```

</td><td>

 ```
0.281
```

</td><td>

 ```
0.260
```

</td></tr>



<tr>
 <td>
 105
</td>
<td>

 ```
en_US/cmu-arctic_low#slp
```

</td><td>

 [wavs/en_US_cmu_arctic_slp.wav](wavs/en_US_cmu_arctic_slp.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_slp.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.136
```

</td><td>

 ```
0.282
```

</td><td>

 ```
0.258
```

</td></tr>



<tr>
 <td>
 106
</td>
<td>

 ```
en_US/cmu-arctic_low#aup
```

</td><td>

 [wavs/en_US_cmu_arctic_aup.wav](wavs/en_US_cmu_arctic_aup.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_aup.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.130
```

</td><td>

 ```
0.282
```

</td><td>

 ```
0.278
```

</td></tr>



<tr>
 <td>
 107
</td>
<td>

 ```
en_US/vctk_low#p363
```

</td><td>

 [wavs/en_US_vctk_p363.wav](wavs/en_US_vctk_p363.wav)

</td><td>

 <img src="./assets/en_US_vctk_p363.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.076
```

</td><td>

 ```
0.285
```

</td><td>

 ```
0.288
```

</td></tr>



<tr>
 <td>
 108
</td>
<td>

 ```
en_US/vctk_low#p232
```

</td><td>

 [wavs/en_US_vctk_p232.wav](wavs/en_US_vctk_p232.wav)

</td><td>

 <img src="./assets/en_US_vctk_p232.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.076
```

</td><td>

 ```
0.285
```

</td><td>

 ```
0.255
```

</td></tr>



<tr>
 <td>
 109
</td>
<td>

 ```
en_US/vctk_low#p247
```

</td><td>

 [wavs/en_US_vctk_p247.wav](wavs/en_US_vctk_p247.wav)

</td><td>

 <img src="./assets/en_US_vctk_p247.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.097
```

</td><td>

 ```
0.285
```

</td><td>

 ```
0.285
```

</td></tr>



<tr>
 <td>
 110
</td>
<td>

 ```
en_US/vctk_low#p300
```

</td><td>

 [wavs/en_US_vctk_p300.wav](wavs/en_US_vctk_p300.wav)

</td><td>

 <img src="./assets/en_US_vctk_p300.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.102
```

</td><td>

 ```
0.287
```

</td><td>

 ```
0.236
```

</td></tr>



<tr>
 <td>
 111
</td>
<td>

 ```
en_US/vctk_low#p254
```

</td><td>

 [wavs/en_US_vctk_p254.wav](wavs/en_US_vctk_p254.wav)

</td><td>

 <img src="./assets/en_US_vctk_p254.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.085
```

</td><td>

 ```
0.290
```

</td><td>

 ```
0.293
```

</td></tr>



<tr>
 <td>
 112
</td>
<td>

 ```
en_US/vctk_low#s5
```

</td><td>

 [wavs/en_US_vctk_s5.wav](wavs/en_US_vctk_s5.wav)

</td><td>

 <img src="./assets/en_US_vctk_s5.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.102
```

</td><td>

 ```
0.290
```

</td><td>

 ```
0.294
```

</td></tr>



<tr>
 <td>
 113
</td>
<td>

 ```
en_US/vctk_low#p302
```

</td><td>

 [wavs/en_US_vctk_p302.wav](wavs/en_US_vctk_p302.wav)

</td><td>

 <img src="./assets/en_US_vctk_p302.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.081
```

</td><td>

 ```
0.291
```

</td><td>

 ```
0.268
```

</td></tr>



<tr>
 <td>
 114
</td>
<td>

 ```
en_US/vctk_low#p286
```

</td><td>

 [wavs/en_US_vctk_p286.wav](wavs/en_US_vctk_p286.wav)

</td><td>

 <img src="./assets/en_US_vctk_p286.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.094
```

</td><td>

 ```
0.292
```

</td><td>

 ```
0.229
```

</td></tr>



<tr>
 <td>
 115
</td>
<td>

 ```
en_US/vctk_low#p288
```

</td><td>

 [wavs/en_US_vctk_p288.wav](wavs/en_US_vctk_p288.wav)

</td><td>

 <img src="./assets/en_US_vctk_p288.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.104
```

</td><td>

 ```
0.292
```

</td><td>

 ```
0.270
```

</td></tr>



<tr>
 <td>
 116
</td>
<td>

 ```
en_US/vctk_low#p287
```

</td><td>

 [wavs/en_US_vctk_p287.wav](wavs/en_US_vctk_p287.wav)

</td><td>

 <img src="./assets/en_US_vctk_p287.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.082
```

</td><td>

 ```
0.292
```

</td><td>

 ```
0.295
```

</td></tr>



<tr>
 <td>
 117
</td>
<td>

 ```
en_US/vctk_low#p345
```

</td><td>

 [wavs/en_US_vctk_p345.wav](wavs/en_US_vctk_p345.wav)

</td><td>

 <img src="./assets/en_US_vctk_p345.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.073
```

</td><td>

 ```
0.293
```

</td><td>

 ```
0.283
```

</td></tr>



<tr>
 <td>
 118
</td>
<td>

 ```
en_US/vctk_low#p333
```

</td><td>

 [wavs/en_US_vctk_p333.wav](wavs/en_US_vctk_p333.wav)

</td><td>

 <img src="./assets/en_US_vctk_p333.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.096
```

</td><td>

 ```
0.293
```

</td><td>

 ```
0.268
```

</td></tr>



<tr>
 <td>
 119
</td>
<td>

 ```
en_US/vctk_low#p244
```

</td><td>

 [wavs/en_US_vctk_p244.wav](wavs/en_US_vctk_p244.wav)

</td><td>

 <img src="./assets/en_US_vctk_p244.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.109
```

</td><td>

 ```
0.294
```

</td><td>

 ```
0.277
```

</td></tr>



<tr>
 <td>
 120
</td>
<td>

 ```
en_US/vctk_low#p250
```

</td><td>

 [wavs/en_US_vctk_p250.wav](wavs/en_US_vctk_p250.wav)

</td><td>

 <img src="./assets/en_US_vctk_p250.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.109
```

</td><td>

 ```
0.295
```

</td><td>

 ```
0.256
```

</td></tr>



<tr>
 <td>
 121
</td>
<td>

 ```
en_US/vctk_low#p299
```

</td><td>

 [wavs/en_US_vctk_p299.wav](wavs/en_US_vctk_p299.wav)

</td><td>

 <img src="./assets/en_US_vctk_p299.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.106
```

</td><td>

 ```
0.298
```

</td><td>

 ```
0.254
```

</td></tr>



<tr>
 <td>
 122
</td>
<td>

 ```
en_US/vctk_low#p260
```

</td><td>

 [wavs/en_US_vctk_p260.wav](wavs/en_US_vctk_p260.wav)

</td><td>

 <img src="./assets/en_US_vctk_p260.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.087
```

</td><td>

 ```
0.298
```

</td><td>

 ```
0.279
```

</td></tr>



<tr>
 <td>
 123
</td>
<td>

 ```
en_US/m-ailabs_low#elliot_miller
```

</td><td>

 [wavs/en_US_m-ailabs_elliot_miller.wav](wavs/en_US_m-ailabs_elliot_miller.wav)

</td><td>

 <img src="./assets/en_US_m-ailabs_elliot_miller.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.113
```

</td><td>

 ```
0.299
```

</td><td>

 ```
0.324
```

</td></tr>



<tr>
 <td>
 124
</td>
<td>

 ```
en_US/vctk_low#p279
```

</td><td>

 [wavs/en_US_vctk_p279.wav](wavs/en_US_vctk_p279.wav)

</td><td>

 <img src="./assets/en_US_vctk_p279.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.092
```

</td><td>

 ```
0.301
```

</td><td>

 ```
0.270
```

</td></tr>



<tr>
 <td>
 125
</td>
<td>

 ```
en_US/cmu-arctic_low#axb
```

</td><td>

 [wavs/en_US_cmu_arctic_axb.wav](wavs/en_US_cmu_arctic_axb.wav)

</td><td>

 <img src="./assets/en_US_cmu-arctic_axb.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.145
```

</td><td>

 ```
0.305
```

</td><td>

 ```
0.279
```

</td></tr>



<tr>
 <td>
 126
</td>
<td>

 ```
en_US/hifi-tts_low#9017
```

</td><td>

 [wavs/en_US_hifi-tts_9017.wav](wavs/en_US_hifi-tts_9017.wav)

</td><td>

 <img src="./assets/en_US_hifi-tts_9017.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.109
```

</td><td>

 ```
0.311
```

</td><td>

 ```
0.281
```

</td></tr>



<tr>
 <td>
 127
</td>
<td>

 ```
en_US/vctk_low#p361
```

</td><td>

 [wavs/en_US_vctk_p361.wav](wavs/en_US_vctk_p361.wav)

</td><td>

 <img src="./assets/en_US_vctk_p361.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.114
```

</td><td>

 ```
0.314
```

</td><td>

 ```
0.274
```

</td></tr>



<tr>
 <td>
 128
</td>
<td>

 ```
en_US/hifi-tts_low#6097
```

</td><td>

 [wavs/en_US_hifi-tts_6097.wav](wavs/en_US_hifi-tts_6097.wav)

</td><td>

 <img src="./assets/en_US_hifi-tts_6097.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.099
```

</td><td>

 ```
0.322
```

</td><td>

 ```
0.324
```

</td></tr>



<tr>
 <td>
 129
</td>
<td>

 ```
en_US/vctk_low#p236
```

</td><td>

 [wavs/en_US_vctk_p236.wav](wavs/en_US_vctk_p236.wav)

</td><td>

 <img src="./assets/en_US_vctk_p236.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.127
```

</td><td>

 ```
0.322
```

</td><td>

 ```
0.267
```

</td></tr>



<tr>
 <td>
 130
</td>
<td>

 ```
en_US/ljspeech_low
```

</td><td>

 [wavs/en_US_ljspeech.wav](wavs/en_US_ljspeech.wav)

</td><td>

 <img src="./assets/en_US_ljspeech.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.155
```

</td><td>

 ```
0.328
```

</td><td>

 ```
0.360
```

</td></tr>



<tr>
 <td>
 131
</td>
<td>

 ```
en_US/hifi-tts_low#92
```

</td><td>

 [wavs/en_US_hifi-tts_92.wav](wavs/en_US_hifi-tts_92.wav)

</td><td>

 <img src="./assets/en_US_hifi-tts_92.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.128
```

</td><td>

 ```
0.330
```

</td><td>

 ```
0.312
```

</td></tr>



<tr>
 <td>
 132
</td>
<td>

 ```
en_US/vctk_low#p248
```

</td><td>

 [wavs/en_US_vctk_p248.wav](wavs/en_US_vctk_p248.wav)

</td><td>

 <img src="./assets/en_US_vctk_p248.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.171
```

</td><td>

 ```
0.337
```

</td><td>

 ```
0.279
```

</td></tr>



<tr>
 <td>
 133
</td>
<td>

 ```
en_UK/apope_low
```

</td><td>

 [wavs/en_UK_apope.wav](wavs/en_UK_apope.wav)

</td><td>

 <img src="./assets/en_UK_apope.png" alt="sounds" width="101" height="101">

</td><td>

 ```
0.129
```

</td><td>

 ```
0.349
```

</td><td>

 ```
0.311
```

</td></tr>



</table>
