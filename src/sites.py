# role: known_analog | saudi | control | background ; r = analog-zone radius (km); coastal = apply water mask
def S(lon, lat, prod, role, r=0, coastal=False, pad=None):
    return dict(lon=lon, lat=lat, prod=prod, role=role, r=r, coastal=coastal, pad=pad)
A, N = "ASTGTM", "NASADEM_HGT"
SITES = {
 "haughton":S(-89.67,75.37,A,"known_analog",15,True,pad=0.6),
 "dry_valleys":S(161.5,-77.5,A,"known_analog",20),
 "askja":S(-16.75,65.03,A,"known_analog",20),
 "atacama":S(-70.0,-24.1,N,"known_analog",40,True),
 "hawaii":S(-155.5,19.65,N,"known_analog",40,True),
 "death_valley":S(-116.9,36.4,N,"known_analog",30),
 "craters_moon":S(-113.5,43.4,N,"known_analog",20),
 "khaybar":S(39.9,25.5,N,"saudi",40),
 "rahat":S(39.9,24.2,N,"saudi",40),
 "wahbah":S(41.14,22.9,N,"saudi",10),
 "asir":S(42.4,18.2,N,"saudi"),
 "tuwayq":S(46.5,24.5,N,"saudi"),
 "rub_al_khali":S(52.0,21.0,N,"saudi"),
 "ctrl_smokies":S(-83.5,35.6,N,"control"),
 "ctrl_plains":S(-98.5,38.5,N,"control"),
 "alps":S(6.87,45.83,N,"background"), "himalaya":S(86.9,27.9,N,"background"),
 "tibesti":S(18.5,19.8,N,"background"), "ahaggar":S(5.5,23.3,N,"background"),
 "sinai":S(33.97,28.5,N,"background"), "grand_canyon":S(-112.1,36.1,N,"background"),
 "andes":S(-70.0,-33.0,N,"background"), "namib":S(15.5,-24.5,N,"background"),
 "gobi":S(100.5,43.5,N,"background"), "etna":S(14.99,37.75,N,"background",0,True),
 "svalbard":S(14.0,78.2,A,"background",0,True), "hekla":S(-19.3,64.7,A,"background"),
}