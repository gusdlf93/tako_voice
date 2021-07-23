# Tako_voice

>Reference : 
>>[토크ON세미나] 딥러닝 기반 음성합성(1)

>Tools :
>>https://github.com/Tyrrrz/YoutubeDownloader


>Process :
>>1. 타겟 동영상 다운로드

>>2. 반복적인 음악이 많이 나오는데, 반복적인 음악 위에다가 랜덤한 음성을 씌워서, denoising autoencode를 구성해서 반복되는 음악도 제거해보기
>>반복되는 음악은 알고 있으니, 음성을 합성해서 만든 다음에, 합성된 음악으로부터 원본의 음악을 예측해내야 함.

>>3. 영상에서 특정 목소리가 있는 부분만을 추출하는 네트워크 학습
>>>반복되는 음악은 지웠으니, 음성이 나오는 부분만을 캡쳐해서 학습.

>>4. 실시간 Voice Conversion Network 만들기 
>>>178개의 동영상을 학습해서 사용하기.

