SMS spam classification

# Source
https://github.com/awasthiabhijeet/Learning-From-Rules/blob/master/data/SMS/

# Labels
0 HAM
1 SPAM

# Labeling Functions

73 rules total, as shown below.

ham	( |^)(thanks\.|thanks)[^\w]*( |$)	Thanks. It was only from tescos but quite nice. All gone now. Speak soon
spam	( |^)(call|ringtone|ringtone)[^\w]* ([^\s]+ )*(free|free)[^\w]*( |$)	Ringtone Club: Get the UK singles chart on your mobile each week and choose any top quality ringtone! This message is free of charge.
ham	( |^)(thats|thats)[^\w]* (\w+ ){0,1}(nice\.|nice)[^\w]*( |$)	Well thats nice. Too bad i cant eat it
spam	( |^)(won|won)[^\w]* ([^\s]+ )*(cash|cash)[^\w]* ([^\s]+ )*(prize!|prize)[^\w]*( |$)	Please call our customer service representative on FREEPHONE 0808 145 4742 between 9am-11pm as you have WON a guaranteed ??1000 cash or ??5000 prize!
spam	( |^)(winner!|winner)[^\w]* ([^\s]+ )*(reward!|reward)[^\w]*( |$)	WINNER! As a valued network customer you hvae been selected to receive a ??900 reward! To collect call 09061701444. Valid 24 hours only. ACL03530150PM
spam	( |^)(guaranteed|guaranteed)[^\w]* ([^\s]+ )*(free|free)[^\w]*( |$)	Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066 TnCs www.Ldew.com1win150ppmx3age16
spam	( |^)(dating|dating)[^\w]* ([^\s]+ )*(call|call)[^\w]*( |$)	Someone has conacted our dating service and entered your phone because they fancy you!To find out who it is call from landline 09111030116. PoBox12n146tf15
ham	( |^)(i)[^\w]* (\w+ ){0,1}(can|can)[^\w]* ([^\s]+ )*(did|did)[^\w]*( |$)	Oh yes I can speak txt 2 u no! Hmm. Did u get  email?
spam	( |^)(guranteed|guaranteed|guaranteed)[^\w]* ([^\s]+ )*(gift\.|gift)[^\w]*( |$)	Great News! Call FREEFONE 08006344447 to claim your guaranteed ??1000 CASH or ??2000 gift. Speak to a live operator NOW!
spam	( |^)(call|call)[^\w]* ([^\s]+ )*(for|for)[^\w]* ([^\s]+ )*(offer|offers|offers\.)[^\w]*( |$)	Double Mins & Double Txt & 1/2 price Linerental on Latest Orange Bluetooth mobiles. Call MobileUpd8 for the very latest offers. 08000839402 or call2optout/LF56
ham	( |^)(can't|can't)[^\w]* (\w+ ){0,1}(talk|talk)[^\w]*( |$)	sry can't talk on phone, with parents
ham	( |^)(i)[^\w]*( |$)	Don know. I did't msg him recently.
ham	( |^)(u)[^\w]* ([^\s]+ )*(how|how)[^\w]* (\w+ ){0,1}(2)[^\w]*( |$)	Do u noe how 2 send files between 2 computers?
ham	( |^)(should|should)[^\w]* (\w+ ){0,1}(i)[^\w]*( |$)	Nah, Wednesday. When should I bring the mini cheetos bag over?
ham	( |^)(your|your)[^\w]* ([^\s]+ )*(i)[^\w]*( |$)	is your hamster dead? Hey so tmr i meet you at 1pm orchard mrt?
ham	( |^)(i)[^\w]* ([^\s]+ )*(miss|miss)[^\w]*( |$)	Goodnight da thangam I really miss u dear.
ham	( |^)(that's|that's)[^\w]* (\w+ ){0,1}(fine!|fine)[^\w]*( |$)	Yeah, that's fine! It's ??6 to get in, is that ok?
spam	( |^)(won|won)[^\w]* ([^\s]+ )*(claim,|claim)[^\w]*( |$)	449050000301 You have won a ??2,000 price! To claim, call 09050000301.
spam	( |^)(welcome!|welcome)[^\w]* ([^\s]+ )*(reply|reply)[^\w]*( |$)	Welcome! Please reply with your AGE and GENDER to begin. e.g 24M
ham	( |^)(mine\.|mine)[^\w]*( |$)	4 oclock at mine. Just to bash out a flat plan.
ham	( |^)(we|we)[^\w]* (\w+ ){0,1}(will|will)[^\w]*( |$)	At 7 we will go ok na.
spam	( |^)(alert|urgent|urgent!)[^\w]* ([^\s]+ )*(award|awarded|awarded)[^\w]* ([^\s]+ )*(guaranteed\.|guaranteed)[^\w]*( |$)	URGENT! Your Mobile number has been awarded with a ??2000 prize GUARANTEED. Call 09058094455 from land line. Claim 3030. Valid 12hrs only
ham	( |^)(do|do)[^\w]* (\w+ ){0,1}(you|you)[^\w]*( |$)	Where do you need to go to get it?
spam	( |^)(no|no)[^\w]* (\w+ ){0,1}(extra|extra)[^\w]*( |$)	SMS SERVICES. for your inclusive text credits, pls goto www.comuk.net login= 3qxj9 unsubscribe with STOP, no extra charge. help 08702840625.COMUK. 220-CM2 9AE
spam	( |^)(unlimited|unlimited)[^\w]* ([^\s]+ )*(calls|calls)[^\w]*( |$)	Freemsg: 1-month unlimited free calls! Activate SmartCall Txt: CALL to No: 68866. Subscriptn3gbp/wk unlimited calls Help: 08448714184 Stop?txt stop landlineonly
spam	( |^)(important|important)[^\w]* ([^\s]+ )*(lucky|lucky)[^\w]*( |$)	IMPORTANT INFORMATION 4 ORANGE USER 0796XXXXXX. TODAY IS UR LUCKY DAY!2 FIND OUT WHY LOG ONTO http://www.urawinner.com THERE'S A FANTASTIC PRIZEAWAITING YOU!
ham	( |^)(i)[^\w]* ([^\s]+ )*(it|it)[^\w]*( |$)	I re-met alex nichols from middle school and it turns out he's dealing!
ham	( |^)(where|where)[^\w]* ([^\s]+ )*(are|are)[^\w]* (\w+ ){0,1}(you|you)[^\w]*( |$)	Where in abj are you serving. Are you staying with dad or alone.
ham	( |^)(my|my)[^\w]* ([^\s]+ )*(kids|kids)[^\w]*( |$)	my ex-wife was not able to have kids. Do you want kids one day?
ham	( |^)(i)[^\w]* (\w+ ){0,1}(used|use|use)[^\w]* (\w+ ){0,1}(to|to)[^\w]*( |$)	Normally i use to drink more water daily:)
spam	( |^)(new|new)[^\w]* (\w+ ){0,1}(mobiles|mobiles)[^\w]* ([^\s]+ )*(only|only)[^\w]*( |$)	500 New Mobiles from 2004, MUST GO! Txt: NOKIA to No: 89545 & collect yours today!From ONLY ??1 www.4-tc.biz 2optout 087187262701.50gbp/mtmsg18
ham	( |^)(did|did)[^\w]* (\w+ ){0,1}(u)[^\w]* (\w+ ){0,1}(got|got)[^\w]*( |$)	Did u got that persons story
spam	( |^)(chat|chat)[^\w]* (\w+ ){0,1}(to|to)[^\w]*( |$)	Dear U've been invited to XCHAT. This is our final attempt to contact u! Txt CHAT to 86688
spam	( |^)(won|won)[^\w]* ([^\s]+ )*(call|call)[^\w]*( |$)	RGENT! This is the 2nd attempt to contact U!U have WON ??1250 CALL 09071512433 b4 050703 T&CsBCM4235WC1N3XX. callcost 150ppm mobilesvary. max??7. 50
spam	( |^)(latest|latest)[^\w]* (\w+ ){0,1}(offer|offers|offers\.)[^\w]*( |$)	Double Mins & Double Txt & 1/2 price Linerental on Latest Orange Bluetooth mobiles. Call MobileUpd8 for the very latest offers. 08000839402 or call2optout/LF56
spam	( |^)(expires|expires)[^\w]* ([^\s]+ )*(now!|now)[^\w]*( |$)	IMPORTANT MESSAGE. This is a final contact attempt. You have important messages waiting out our customer claims dept. Expires 13/4/04. Call 08717507382 NOW!
spam	( |^)(win|win)[^\w]* ([^\s]+ )*(shopping|shopping)[^\w]*( |$)	WIN a ??200 Shopping spree every WEEK Starting NOW. 2 play text STORE to 88039. SkilGme. TsCs08714740323 1Winawk! age16 ??1.50perweeksub.
ham	( |^)(i'll|i'll)[^\w]*( |$)	Yar else i'll thk of all sorts of funny things.
spam	( |^)(chat|chat)[^\w]* ([^\s]+ )*(date|date)[^\w]*( |$)	Bored housewives! Chat n date now! 0871750.77.11! BT-national rate 10p/min only from landlines!
spam	( |^)(please|please)[^\w]* (\w+ ){0,1}(call|call)[^\w]* ([^\s]+ )*(service|service)[^\w]*( |$)	Please call our customer service representative on FREEPHONE 0808 145 4742 between 9am-11pm as you have WON a guaranteed ??1000 cash or ??5000 prize!
spam	( |^)(free\.|free)[^\w]* ([^\s]+ )*(sex|sex)[^\w]*( |$)	This message is free. Welcome to the new & improved Sex & Dogging club! To unsubscribe from this service reply STOP. msgs@150p 18 only
spam	( |^)(free|free)[^\w]* ([^\s]+ )*(price|price)[^\w]* ([^\s]+ )*(call|call)[^\w]*( |$)	Free video camera phones with Half Price line rental for 12 mths and 500 cross ntwk mins 100 txts. Call MobileUpd8 08001950382 or Call2OptOut/674
spam	( |^)(cash|cash)[^\w]* ([^\s]+ )*(prize|prize|prize!)[^\w]*( |$)	You have won ?1,000 cash or a ?2,000 prize! To claim, call09050000327. T&C: RSTM, SW7 3SS. 150ppm
ham	( |^)(fb|fb)[^\w]*( |$)	Friends that u can stay on fb chat with
spam	( |^)(free|free)[^\w]* ([^\s]+ )*(phone|phones|phones)[^\w]*( |$)	Free video camera phones with Half Price line rental for 12 mths and 500 cross ntwk mins 100 txts. Call MobileUpd8 08001950382 or Call2OptOut/674&
ham	( |^)(noisy\.|noisy)[^\w]*( |$)	Mine here like all fr china then so noisy.
ham	( |^)(adventuring|adventuring)[^\w]*( |$)	happened here while you were adventuring
spam	( |^)(password|password)[^\w]*( |$)	Send me your id and password
ham	( |^)(maggi|maggi)[^\w]*( |$)	No need to buy lunch for me.. I eat maggi mee..
ham	( |^)(wtf\.|wtf)[^\w]*( |$)	&lt;#&gt; ISH MINUTES WAS 5 MINUTES AGO. WTF.
spam	( |^)(won|won)[^\w]* ([^\s]+ )*(cash|cash)[^\w]*( |$)	Please call our customer service representative on FREEPHONE 0808 145 4742 between 9am-11pm as you have WON a guaranteed ??1000 cash or ??5000 prize!
ham	( |^)(amrita|amrita)[^\w]*( |$)	Staff of placement training in Amrita college.
ham	( |^)(praying|praying\.will|praying\.will)[^\w]*( |$)	I am joining today formally.Pls keep praying.will talk later.
spam	( |^)(childporn|childporn)[^\w]*( |$)	Ic. There are a lotta childporn cars then.
ham	( |^)(shit|shit)[^\w]*( |$)	Just wanted to say holy shit you guys weren't kidding about this bud
spam	( |^)(credits|credits)[^\w]*( |$)	SMS SERVICES For your inclusive text credits pls gotto www.comuk.net login 3qxj9 unsubscribe with STOP no extra charge help 08702840625 comuk.220cm2 9AE
ham	( |^)(goodo!|goodo)[^\w]* ([^\s]+ )*(we|we)[^\w]*( |$)	Goodo! Yes we must speak friday - egg-potato ratio for tortilla needed!
spam	( |^)(latest|latest)[^\w]*( |$)	Double Mins & Double Txt & 1/2 price Linerental on Latest Orange Bluetooth mobiles. Call MobileUpd8 for the very latest offers. 08000839402 or call2optout/LF56
spam	( |^)(\?\?5000|\?\?5000)[^\w]* ([^\s]+ )*(09050090044|09050090044)[^\w]*( |$)	WELL DONE! Your 4* Costa Del Sol Holiday or ??5000 await collection. Call 09050090044 Now toClaim. SAE, TCs, POBox334, Stockport, SK38xh, Cost??1.50/pm, Max10mins
spam	( |^)(hard|hard)[^\w]* (\w+ ){0,1}(live|live)[^\w]* ([^\s]+ )*(girl|girl)[^\w]*( |$)	Hard LIVE 121 chat just 60p/min. Choose your girl and connect LIVE. Call 09094646899 now! Cheap Chat UK's biggest live service. VU BCM1896WC1N3XX
ham	( |^)(link|link)[^\w]*( |$)	A link to your picture has been sent. You can also use http://alto18.co.uk/wave/wave.asp?o=44345
spam	( |^)(urgent|urgent)[^\w]* ([^\s]+ )*(prize|prize)[^\w]*( |$)	URGENT This is our 2nd attempt to contact U. Your ??900 prize from YESTERDAY is still awaiting collection. To claim CALL NOW 09061702893. ACL03530150PM
spam	( |^)(sms\.|sms)[^\w]* ([^\s]+ )*(reply|reply)[^\w]*( |$)	SMS. ac Sptv: The New Jersey Devils and the Detroit Red Wings play Ice Hockey. Correct or Incorrect? End? Reply END SPTV
spam	( |^)(direct|direct)[^\w]*( |$)	Call Germany for only 1 pence per minute! Call from a fixed line via access number 0844 861 85 85. No prepayment. Direct access!
spam	( |^)(voucher|voucher)[^\w]* ([^\s]+ )*(claim|claim)[^\w]*( |$)	Dear Voucher Holder, To claim this weeks offer, at you PC please go to http://www.e-tlp.co.uk/reward. Ts&Cs apply.
spam	( |^)(\?\?1\.50|\?\?1\.50)[^\w]*( |$)	Oh my god! I've found your number again! I'm so glad, text me back xafter this msgs cst std ntwk chg ??1.50
ham	( |^)(hee|hee)[^\w]*( |$)	They can try! They can get lost, in fact. Tee hee
ham	( |^)(jus|jus)[^\w]*( |$)	Jus finished avatar nigro
spam	( |^)(free|free)[^\w]* ([^\s]+ )*(tone|tone)[^\w]*( |$)	FREE RING TONE just text \POLYS\" to 87131. Then every week get a new tone. 0870737910216yrs only ??1.50/wk."
spam	( |^)(message\.|message)[^\w]* ([^\s]+ )*(call|call)[^\w]*( |$)	You have 1 new message. Please call 08715205273
spam	( |^)(fantasies\.|fantasies)[^\w]* (\w+ ){0,1}(call|call)[^\w]*( |$)	I'd like to tell you my deepest darkest fantasies. Call me 09094646631 just 60p/min. To stop texts call 08712460324 (nat rate)
spam	( |^)(\?\?500|\?\?500)[^\w]*( |$)	Ur HMV Quiz cash-balance is currently ??500 - to maximize ur cash-in now send HMV1 to 86688 only 150p/msg
spam	( |^)(inviting|inviting)[^\w]* ([^\s]+ )*(friend\.|friend)[^\w]*( |$)	Natalie (20/F) is inviting you to be her friend. Reply YES-165 or NO-165 See her: www.SMS.ac/u/natalie2k9 STOP? Send STOP FRND to 62468