# Tako_voice

>Reference : 
>>[토크ON세미나] 딥러닝 기반 음성합성(1)

>Tools :
>>https://github.com/Tyrrrz/YoutubeDownloader

>Process :
>>1. 타겟 동영상 다운로드

>>2. Denoising AutoEncoder 1 
>>반복적인 음악이 많이 나오는데, 반복적인 음악 위에다가 랜덤한 음성 데이터셋을 씌워서 음악을 제거하는 모델을 구성
>> denoising autoencoder를 구성해서 배경 음악(Noise) + 음성 = 음성 이 되도록 학습
>> denoising autoencoder를 테스트할 때, 그림만 그리는(노래만 나오는)동영상에서 목소리들을 가져오기.

>>3. Enhanced Denosing AutoEncoder 2
>> 특정 대상의 목소리만 구했으면, 이제 해당목소리에 다른 목소리랑, 음악, 랜덤한 영상의 음악들을 다 들고와서 합친다음에, 원본 목소리로 복원
>> Denoising autoencoder로 여러개의 음악과 목소리를 합친 다음에, 원본 목소리만 복원
>> Test에는 전체 178개 동영상을 대상으로 목소리들을 가져오기.

>>4. 실시간 Voice Conversion Network 만들기 
>>>178개의 동영상을 학습해서 사용하기.
