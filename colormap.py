
from matplotlib.colors import ListedColormap

cm_type = "linear"

cm_data = [[1.00000000,1.00000000,1.00000000],
           [0.99385494,0.99656861,0.98714573],
           [0.98817164,0.99205112,0.9829924 ],
           [0.98250093,0.98755253,0.97885099],
           [0.97683963,0.98307355,0.97472483],
           [0.97118713,0.97861405,0.97061516],
           [0.96554476,0.97417335,0.96652168],
           [0.95991083,0.96975171,0.96244603],
           [0.95428565,0.9653488 ,0.95838835],
           [0.94867018,0.96096407,0.95434823],
           [0.94306313,0.95659771,0.95032673],
           [0.9374662 ,0.95224897,0.94632279],
           [0.93187685,0.94791843,0.94233818],
           [0.92629698,0.9436053 ,0.93837161],
           [0.92072621,0.93930949,0.93442334],
           [0.91516471,0.93503077,0.93049316],
           [0.90961222,0.93076902,0.92658118],
           [0.90406886,0.92652402,0.92268722],
           [0.89853462,0.92229558,0.91881114],
           [0.89300819,0.91808395,0.91495369],
           [0.88749096,0.9138885 ,0.91111378],
           [0.88198348,0.90970889,0.90729091],
           [0.87648352,0.90554564,0.90348637],
           [0.87099253,0.90139812,0.89969904],
           [0.86551061,0.89726614,0.89592867],
           [0.86003609,0.89315003,0.89217617],
           [0.85457116,0.88904897,0.88843992],
           [0.84911317,0.8849636 ,0.88472146],
           [0.84366422,0.8808931 ,0.88101922],
           [0.83822294,0.87683773,0.8773339 ],
           [0.83278941,0.87279731,0.87366525],
           [0.82736478,0.86877132,0.87001231],
           [0.82194615,0.86476049,0.86637677],
           [0.81653564,0.860764  ,0.86275706],
           [0.81113379,0.85678154,0.85915262],
           [0.80573746,0.8528139 ,0.85556525],
           [0.80034864,0.84886031,0.85199349],
           [0.7949676 ,0.84492054,0.84843695],
           [0.78959389,0.84099458,0.84489572],
           [0.7842252 ,0.83708294,0.84137104],
           [0.7788635 ,0.83318488,0.83786147],
           [0.77350855,0.82930034,0.83436696],
           [0.76816021,0.82542918,0.83088736],
           [0.7628183 ,0.82157131,0.82742261],
           [0.75748169,0.81772691,0.82397319],
           [0.75215049,0.81389578,0.82053883],
           [0.74682508,0.81007768,0.81711907],
           [0.74150527,0.80627249,0.81371383],
           [0.73619088,0.80248012,0.810323  ],
           [0.73088162,0.79870049,0.80694656],
           [0.72557729,0.79493353,0.80358442],
           [0.72027766,0.79117912,0.80023651],
           [0.71498282,0.78743711,0.79690255],
           [0.70969227,0.78370747,0.79358265],
           [0.70440576,0.77999013,0.79027673],
           [0.69912329,0.77628494,0.78698459],
           [0.69384467,0.77259179,0.78370612],
           [0.68856939,0.76891068,0.78044143],
           [0.68329756,0.76524141,0.77719021],
           [0.67802886,0.76158394,0.77395245],
           [0.67276301,0.75793817,0.77072809],
           [0.66749999,0.75430397,0.76751692],
           [0.66223925,0.75068133,0.76431905],
           [0.656981  ,0.74707003,0.76113413],
           [0.65172454,0.74347012,0.75796235],
           [0.64647018,0.73988135,0.75480329],
           [0.64121728,0.73630375,0.75165711],
           [0.63596572,0.73273718,0.74852365],
           [0.63071541,0.72918151,0.74540271],
           [0.62546575,0.72563674,0.74229444],
           [0.62021703,0.72210265,0.7391984 ],
           [0.61496858,0.71857924,0.73611476],
           [0.60971926,0.71506661,0.7330444 ],
           [0.60447787,0.71156145,0.72999006],
           [0.5992416 ,0.70806435,0.72695328],
           [0.59400935,0.70457546,0.72393428],
           [0.58878064,0.70109474,0.72093339],
           [0.58355619,0.69762181,0.71795022],
           [0.57833536,0.69415668,0.71498524],
           [0.57311874,0.69069901,0.71203816],
           [0.56790594,0.68724872,0.70910928],
           [0.56269737,0.68380554,0.70619844],
           [0.55749261,0.68036938,0.703306  ],
           [0.55229225,0.67693993,0.70043168],
           [0.5470958 ,0.67351712,0.69757589],
           [0.54190398,0.6701006 ,0.69473832],
           [0.53671636,0.66669029,0.69191935],
           [0.53153327,0.66328594,0.68911892],
           [0.52635501,0.65988728,0.68633696],
           [0.52118127,0.65649422,0.68357385],
           [0.51601259,0.65310644,0.6808294 ],
           [0.5108492 ,0.64972371,0.67810362],
           [0.50569113,0.64634584,0.6753967 ],
           [0.50053845,0.64297262,0.67270879],
           [0.49539166,0.63960378,0.67003976],
           [0.49025103,0.63623905,0.66738966],
           [0.48511684,0.63287821,0.66475853],
           [0.47999088,0.6295206 ,0.66214628],
           [0.4748735 ,0.62616597,0.65955296],
           [0.46976396,0.62281437,0.65697863],
           [0.46466269,0.61946554,0.65442326],
           [0.45957007,0.61611921,0.65188692],
           [0.45448676,0.61277509,0.64936944],
           [0.44941321,0.60943289,0.64687085],
           [0.44435406,0.6060913 ,0.64439065],
           [0.43930711,0.60275076,0.64192899],
           [0.4342723 ,0.59941118,0.63948579],
           [0.42925039,0.59607225,0.63706087],
           [0.42424275,0.59273352,0.634654  ],
           [0.41925524,0.58939341,0.63226462],
           [0.41428357,0.58605294,0.62989278],
           [0.40932876,0.58271178,0.62753814],
           [0.40439308,0.5793693 ,0.62520036],
           [0.39948237,0.57602404,0.6228786 ],
           [0.39459211,0.57267707,0.62057281],
           [0.3897232 ,0.56932812,0.61828273],
           [0.38488343,0.56597536,0.61600709],
           [0.38007   ,0.56261945,0.61374571],
           [0.37528202,0.5592606 ,0.61149822],
           [0.37052667,0.55589719,0.60926332],
           [0.36580317,0.55252944,0.60704066],
           [0.36110952,0.54915788,0.60482956],
           [0.3564534 ,0.54578081,0.60262889],
           [0.35183333,0.54239869,0.60043789],
           [0.3472471 ,0.53901208,0.59825613],
           [0.34270457,0.53561892,0.59608189],
           [0.33820047,0.53222049,0.59391488],
           [0.33373433,0.52881701,0.59175431],
           [0.32931744,0.52540617,0.58959837],
           [0.32494043,0.52199014,0.58744705],
           [0.32060687,0.51856829,0.58529921],
           [0.31632321,0.51513942,0.58315342],
           [0.31208195,0.51170525,0.5810096 ],
           [0.3078883 ,0.50826484,0.57886637],
           [0.3037445 ,0.50481788,0.5767229 ],
           [0.29964475,0.50136571,0.5745788 ],
           [0.29559448,0.49790737,0.57243303],
           [0.29159409,0.49444293,0.57028472],
           [0.28763829,0.49097356,0.56813382],
           [0.28373109,0.48749861,0.56597925],
           [0.27987438,0.48401782,0.56382036],
           [0.27606191,0.48053251,0.56165722],
           [0.2722938 ,0.47704278,0.55948913],
           [0.26857818,0.47354715,0.55731517],
           [0.26490566,0.47004749,0.55513575],
           [0.2612758 ,0.46654399,0.55295029],
           [0.25769007,0.4630364 ,0.55075862],
           [0.25415186,0.45952415,0.54856013],
           [0.25065434,0.45600856,0.54635519],
           [0.24719686,0.45248981,0.5441434 ],
           [0.24377874,0.44896802,0.5419249 ],
           [0.24040405,0.44544248,0.5396991 ],
           [0.23706877,0.44191392,0.5374663 ],
           [0.23377033,0.43838282,0.53522658],
           [0.23050795,0.43484929,0.53297997],
           [0.22728084,0.43131348,0.53072643],
           [0.2240882 ,0.42777548,0.52846604],
           [0.22093212,0.42423491,0.52619864],
           [0.21781021,0.42069215,0.52392442],
           [0.21472003,0.41714759,0.52164362],
           [0.21166081,0.41360129,0.51935637],
           [0.20863173,0.41005335,0.51706277],
           [0.20563192,0.40650386,0.51476279],
           [0.20266068,0.40295287,0.51245672],
           [0.19971723,0.39940042,0.51014471],
           [0.19680084,0.39584656,0.50782693],
           [0.19391066,0.39229135,0.50550343],
           [0.19104594,0.38873483,0.50317433],
           [0.1882061 ,0.38517698,0.50084003],
           [0.18539024,0.38161788,0.49850042],
           [0.18259768,0.37805752,0.49615573],
           [0.17982794,0.37449584,0.49380638],
           [0.17707992,0.37093298,0.49145204],
           [0.1743534 ,0.36736878,0.48909339],
           [0.17164735,0.36380333,0.48673017],
           [0.16896142,0.36023653,0.48436288],
           [0.16629477,0.35666841,0.48199144],
           [0.16364704,0.35309885,0.47961626],
           [0.16101728,0.34952791,0.47723716],
           [0.1584054 ,0.34595538,0.47485478],
           [0.15581019,0.34238141,0.47246868],
           [0.15323162,0.33880575,0.47007946],
           [0.1506691 ,0.33522836,0.46768726],
           [0.14812175,0.33164924,0.46529192],
           [0.14559418,0.32806688,0.46289773],
           [0.14305981,0.32448483,0.46050859],
           [0.14052105,0.32090216,0.45812652],
           [0.13797752,0.31731876,0.45575124],
           [0.1354295 ,0.31373428,0.45338304],
           [0.13287686,0.31014851,0.45102185],
           [0.13031936,0.30656124,0.44866753],
           [0.1277575 ,0.30297207,0.44632057],
           [0.12519064,0.29938088,0.44398046],
           [0.12261937,0.29578722,0.4416478 ],
           [0.12004302,0.29219097,0.43932207],
           [0.11746228,0.28859163,0.43700392],
           [0.11487642,0.28498907,0.43469284],
           [0.1122862 ,0.28138276,0.43238951],
           [0.10969101,0.2777725 ,0.43009352],
           [0.10709109,0.27415789,0.42780516],
           [0.1044866 ,0.27053848,0.42552463],
           [0.10187723,0.26691399,0.42325178],
           [0.09926341,0.26328389,0.42098705],
           [0.09664509,0.25964778,0.41873049],
           [0.09402217,0.25600524,0.41648212],
           [0.09139478,0.25235579,0.41424213],
           [0.08876338,0.24869882,0.41201098],
           [0.08612786,0.24503388,0.40978868],
           [0.08348846,0.24136039,0.40757551],
           [0.08084536,0.23767775,0.40537166],
           [0.07819885,0.23398534,0.40317745],
           [0.07554858,0.23028266,0.40099268],
           [0.07289449,0.22656912,0.39881733],
           [0.07023791,0.22284371,0.39665248],
           [0.06757945,0.21910561,0.3944986 ],
           [0.06491969,0.21535395,0.39235609],
           [0.06225938,0.21158782,0.39022546],
           [0.05959752,0.20780669,0.38810589],
           [0.05693524,0.20400948,0.38599809],
           [0.05427546,0.20019457,0.38390396],
           [0.05161918,0.19636083,0.38182393],
           [0.04896541,0.19250757,0.37975698],
           [0.04631656,0.1886332 ,0.37770436],
           [0.0436765 ,0.18473573,0.37566812],
           [0.04104588,0.18081392,0.37364799],
           [0.03842152,0.17686643,0.37164376],
           [0.03590282,0.17289021,0.36965919],
           [0.03349794,0.16888443,0.36769202],
           [0.03121107,0.16484582,0.36574578],
           [0.02904182,0.16077203,0.36382109],
           [0.0269897 ,0.15666056,0.36191849],
           [0.02505715,0.15250774,0.36004063],
           [0.0232438 ,0.14831042,0.35818831],
           [0.0215506 ,0.14406465,0.35636327],
           [0.01998128,0.13976513,0.35456943],
           [0.01853178,0.13540861,0.352805  ],
           [0.01720819,0.13098781,0.35107588],
           [0.016012  ,0.12649621,0.34938468],
           [0.01494274,0.12192713,0.34773257],
           [0.01400293,0.11727178,0.34612312],
           [0.01319628,0.11251948,0.34456108],
           [0.01252535,0.10765837,0.34305043],
           [0.01199372,0.10267405,0.34159616],
           [0.01160445,0.09754975,0.34020301],
           [0.01136225,0.09226454,0.33887735],
           [0.01126984,0.08679416,0.33762397],
           [0.01133738,0.08110432,0.33645452],
           [0.01156659,0.07515731,0.33537339],
           [0.01196973,0.06889766,0.33439476],
           [0.0125531 ,0.06225662,0.33352768],
           [0.01332742,0.05513691,0.33278557],
           [0.01430852,0.04739538,0.3321871 ],
           [0.01550826,0.03881168,0.33174742],
           [0.01694571,0.02981098,0.33148894],
           [0.01865031,0.02084333,0.33144462],
           [0.02069442,0.01183243,0.33169097]]
colormap = ListedColormap(cm_data)
