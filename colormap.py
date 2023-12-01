
from matplotlib.colors import ListedColormap

cm_type = "linear"

cm_data = [[1.00000000,1.00000000,1.00000000],
           [0.99368993,0.99664034,0.98707652],
           [0.98891058,0.99198492,0.98083536],
           [0.98413241,0.98735749,0.97456409],
           [0.97936331,0.98275629,0.9682473 ],
           [0.97461062,0.97817784,0.96188884],
           [0.9698815 ,0.9736204 ,0.95547601],
           [0.9651817 ,0.96908085,0.94901492],
           [0.96051598,0.96455724,0.94250432],
           [0.95588786,0.96004783,0.9359454 ],
           [0.95129912,0.95555053,0.92934952],
           [0.94675136,0.95106483,0.92271354],
           [0.94224469,0.94658963,0.91604549],
           [0.93777909,0.94212459,0.90934662],
           [0.93335316,0.93766894,0.90262792],
           [0.92896713,0.93322324,0.89588066],
           [0.92461903,0.92878692,0.88911599],
           [0.92030737,0.92435994,0.88233818],
           [0.91603098,0.91994242,0.87554837],
           [0.91178892,0.91553458,0.86874596],
           [0.90758022,0.91113658,0.86193109],
           [0.90340316,0.90674835,0.85510926],
           [0.89925663,0.90236999,0.8482816 ],
           [0.8951395 ,0.89800158,0.84144981],
           [0.89105095,0.89364326,0.83461389],
           [0.88699004,0.88929508,0.82777488],
           [0.8829559 ,0.8849571 ,0.82093371],
           [0.87894771,0.88062935,0.81409129],
           [0.87496471,0.87631188,0.80724837],
           [0.87100615,0.87200469,0.80040584],
           [0.86707145,0.8677078 ,0.79356396],
           [0.86315999,0.86342121,0.78672323],
           [0.85927215,0.85914502,0.77987911],
           [0.85540658,0.85487911,0.77303658],
           [0.85156265,0.85062342,0.76619662],
           [0.84774002,0.84637795,0.75935893],
           [0.84393793,0.84214264,0.75252546],
           [0.840156  ,0.83791743,0.74569623],
           [0.83639455,0.83370234,0.73886801],
           [0.83265388,0.82949734,0.73203769],
           [0.82893227,0.82530228,0.72521237],
           [0.82522917,0.82111708,0.71839329],
           [0.82154646,0.81694089,0.71157989],
           [0.81790247,0.81276584,0.70477689],
           [0.81429657,0.80859289,0.6979744 ],
           [0.81072687,0.80442256,0.69117268],
           [0.80719376,0.80025455,0.68437123],
           [0.80369831,0.79608808,0.67757157],
           [0.80024445,0.79192117,0.67077623],
           [0.79682743,0.78775572,0.66398278],
           [0.79344746,0.78359142,0.65719126],
           [0.79010459,0.77942799,0.65040235],
           [0.78680048,0.77526444,0.64361745],
           [0.78353744,0.77109945,0.63683847],
           [0.78031171,0.76693452,0.63006312],
           [0.77712318,0.76276938,0.62329232],
           [0.77397203,0.75860375,0.61652605],
           [0.77085807,0.7544374 ,0.60976525],
           [0.76778626,0.75026784,0.60301309],
           [0.76475162,0.74609702,0.59626757],
           [0.76175404,0.74192474,0.58952869],
           [0.7587934 ,0.73775078,0.58279701],
           [0.75586943,0.73357493,0.57607341],
           [0.75298278,0.72939665,0.56935841],
           [0.75013633,0.72521427,0.56265518],
           [0.74732598,0.72102939,0.55596144],
           [0.74455146,0.71684184,0.54927777],
           [0.7418124 ,0.71265145,0.54260497],
           [0.73910862,0.70845802,0.53594328],
           [0.73643967,0.70426142,0.52929352],
           [0.73380835,0.70006   ,0.5226587 ],
           [0.73121198,0.6958547 ,0.51603755],
           [0.72864914,0.69164585,0.50943012],
           [0.7261193 ,0.68743335,0.5028371 ],
           [0.723622  ,0.68321709,0.49625899],
           [0.72115668,0.67899699,0.48969645],
           [0.71872274,0.67477299,0.48315009],
           [0.7163233 ,0.67054323,0.4766232 ],
           [0.71395418,0.66630937,0.47011383],
           [0.71161451,0.66207146,0.46362242],
           [0.70930362,0.65782948,0.45714951],
           [0.70702078,0.65358341,0.45069567],
           [0.70476525,0.64933327,0.44426145],
           [0.70253629,0.64507905,0.43784734],
           [0.70033644,0.64081915,0.43145652],
           [0.698162  ,0.636555  ,0.42508719],
           [0.69601173,0.63228688,0.41873945],
           [0.69388471,0.62801486,0.41241385],
           [0.69178016,0.62373898,0.40611073],
           [0.68969721,0.61945933,0.39983046],
           [0.68763492,0.61517602,0.3935735 ],
           [0.68559607,0.61088728,0.38734304],
           [0.68357648,0.60659488,0.38113671],
           [0.68157483,0.60229911,0.37495474],
           [0.67959035,0.59800006,0.36879717],
           [0.67762218,0.59369782,0.36266413],
           [0.67566912,0.58939263,0.35655623],
           [0.67373205,0.58508375,0.35047449],
           [0.67181113,0.58077074,0.34441998],
           [0.66990269,0.57645516,0.33839089],
           [0.66800603,0.57213708,0.33238692],
           [0.6661201 ,0.56781671,0.32640826],
           [0.66424412,0.56349419,0.32045472],
           [0.66237875,0.55916884,0.31452745],
           [0.6605242 ,0.55484018,0.30862811],
           [0.6586772 ,0.55050975,0.30275328],
           [0.65683654,0.54617783,0.2969032 ],
           [0.65500157,0.54184454,0.29107736],
           [0.65317136,0.53751008,0.28527554],
           [0.65134909,0.53317241,0.27950097],
           [0.64953087,0.5288334 ,0.27375033],
           [0.64771492,0.52449373,0.26802269],
           [0.64590081,0.52015343,0.26231706],
           [0.64408753,0.51581277,0.25663316],
           [0.64227837,0.51146963,0.25097413],
           [0.64046969,0.50712581,0.24533659],
           [0.63865987,0.50278198,0.2397185 ],
           [0.6368481 ,0.49843838,0.23411916],
           [0.63503433,0.49409477,0.22853844],
           [0.63322273,0.48974858,0.22297818],
           [0.63140351,0.4854045 ,0.21744379],
           [0.62958254,0.48105955,0.21192775],
           [0.62776022,0.47671321,0.20642959],
           [0.62593295,0.47236669,0.2009602 ],
           [0.6241014 ,0.46801937,0.19551954],
           [0.62226571,0.46367125,0.19010038],
           [0.6204264 ,0.45932181,0.18470217],
           [0.61858219,0.4549714 ,0.17932666],
           [0.61673252,0.45062002,0.17397469],
           [0.61487483,0.44626842,0.1686573 ],
           [0.61300926,0.44191647,0.16336974],
           [0.6111363 ,0.43756382,0.15810833],
           [0.6092557 ,0.43321033,0.15287362],
           [0.60736596,0.42885654,0.14766764],
           [0.60546634,0.4245026 ,0.14249153],
           [0.60355618,0.42014864,0.13734638],
           [0.60163461,0.41579489,0.13223353],
           [0.59970085,0.41144156,0.12715424],
           [0.59775292,0.40708944,0.12211501],
           [0.59579033,0.40273867,0.11711484],
           [0.59381271,0.39838939,0.11215221],
           [0.59181907,0.39404197,0.1072287 ],
           [0.58980836,0.38969688,0.10234599],
           [0.58777947,0.38535464,0.09750586],
           [0.58573129,0.38101576,0.09271008],
           [0.58366275,0.37668082,0.08796045],
           [0.58157272,0.3723504 ,0.08325884],
           [0.57946019,0.36802508,0.07860707],
           [0.57732396,0.36370555,0.07400714],
           [0.57516283,0.35939256,0.06946108],
           [0.57297572,0.35508682,0.06497085],
           [0.57076153,0.35078908,0.06053847],
           [0.56851921,0.34650007,0.05616595],
           [0.56624781,0.34222052,0.05185528],
           [0.56394628,0.33795125,0.04760852],
           [0.56161371,0.333693  ,0.0434277 ],
           [0.55924925,0.32944652,0.03930509],
           [0.55685214,0.32521253,0.03538216],
           [0.55442163,0.32099176,0.03176132],
           [0.55195714,0.31678486,0.02842781],
           [0.54945813,0.31259245,0.02536703],
           [0.54692418,0.30841512,0.0225646 ],
           [0.54435495,0.30425339,0.0200064 ],
           [0.54175012,0.30010781,0.01767852],
           [0.53910961,0.29597872,0.01556749],
           [0.53643336,0.2918665 ,0.01366016],
           [0.5337214 ,0.28777145,0.01194376],
           [0.53097383,0.28369379,0.01040596],
           [0.52819084,0.27963371,0.00903485],
           [0.52537271,0.2755913 ,0.00781898],
           [0.52251979,0.27156661,0.00674741],
           [0.51963246,0.26755964,0.00580963],
           [0.51671117,0.26357031,0.0049956 ],
           [0.51375641,0.25959853,0.00429576],
           [0.51076866,0.25564415,0.00370097],
           [0.50774858,0.25170691,0.00320275],
           [0.50469677,0.24778652,0.00279302],
           [0.50161385,0.24388268,0.00246411],
           [0.49850043,0.2399951 ,0.00220869],
           [0.49535716,0.23612337,0.00201997],
           [0.49218457,0.23226723,0.00189118],
           [0.48898345,0.22842611,0.00181662],
           [0.48575296,0.22460059,0.0018013 ],
           [0.48249469,0.22078954,0.00183267],
           [0.47920948,0.21699242,0.00190193],
           [0.47589831,0.21320834,0.00200553],
           [0.47256173,0.20943681,0.00213895],
           [0.46920003,0.20567763,0.00229696],
           [0.46581593,0.20192734,0.00249318],
           [0.46241296,0.19818382,0.00268562],
           [0.45899419,0.19444338,0.00287961],
           [0.45555911,0.19070609,0.00307495],
           [0.45210751,0.1869717 ,0.00327259],
           [0.4486386 ,0.18324063,0.00347128],
           [0.44515194,0.17951288,0.00367125],
           [0.44164675,0.17578882,0.00387251],
           [0.43812148,0.17206972,0.00407476],
           [0.43457718,0.16835389,0.00427849],
           [0.4310133 ,0.16464143,0.00448369],
           [0.42742932,0.16093242,0.00469054],
           [0.42382464,0.15722697,0.00489903],
           [0.42019868,0.15352519,0.00510934],
           [0.41655077,0.14982731,0.00532132],
           [0.41288032,0.14613342,0.00553535],
           [0.40918666,0.14244373,0.00575148],
           [0.405469  ,0.13875859,0.00596949],
           [0.4017266 ,0.13507831,0.00618938],
           [0.39795892,0.13140286,0.00641233],
           [0.39416409,0.12773412,0.00663678],
           [0.39034059,0.12407342,0.00686357],
           [0.3864887 ,0.12041979,0.00709262],
           [0.38260765,0.1167736 ,0.00732485],
           [0.37869641,0.11313558,0.00756021],
           [0.37475395,0.10950652,0.00779871],
           [0.37077494,0.10589364,0.0080386 ],
           [0.36676154,0.10229328,0.0082816 ],
           [0.36271307,0.09870592,0.00852802],
           [0.35862443,0.09513895,0.00877609],
           [0.35449512,0.0915929 ,0.00902669],
           [0.35032553,0.08806699,0.00927983],
           [0.34610753,0.08457446,0.00953235],
           [0.34184628,0.08110708,0.00978645],
           [0.33753128,0.07768297,0.01003787],
           [0.33316725,0.07429499,0.01028748],
           [0.32874678,0.07095719,0.01053121],
           [0.32426913,0.06767259,0.01076692],
           [0.31973353,0.06444448,0.01099181],
           [0.3151349 ,0.06128478,0.01120016],
           [0.31047238,0.0581981 ,0.01138763],
           [0.30574663,0.05518648,0.01154899],
           [0.30095718,0.05225422,0.01167851],
           [0.29610476,0.04940362,0.01176978],
           [0.29118901,0.04663915,0.01181636],
           [0.28621406,0.04395587,0.01181367],
           [0.28118401,0.04134822,0.01175795],
           [0.27610297,0.03880199,0.0116466 ],
           [0.27097549,0.03638851,0.01147858],
           [0.2658021 ,0.03412114,0.01125671],
           [0.2606102 ,0.03193427,0.01101042],
           [0.25541072,0.02980267,0.01074932],
           [0.25020316,0.02772649,0.0104738 ],
           [0.24498683,0.02570611,0.01018452],
           [0.23976095,0.02374204,0.00988227],
           [0.23452437,0.02183528,0.00956833],
           [0.22927619,0.01998632,0.00924354],
           [0.224015  ,0.01819632,0.0089094 ],
           [0.2187412 ,0.01646377,0.00856466],
           [0.21345152,0.0147923 ,0.00821346],
           [0.20814662,0.01317988,0.00785402],
           [0.20282391,0.01162886,0.00748919],
           [0.1974819 ,0.01013992,0.00712008],
           [0.19211956,0.00871304,0.00674707],
           [0.18673453,0.00734969,0.00637215],
           [0.18132445,0.0060511 ,0.00599708],
           [0.17588734,0.00481785,0.00562292],
           [0.17040702,0.00366538,0.00526667]]
colormap = ListedColormap(cm_data)
