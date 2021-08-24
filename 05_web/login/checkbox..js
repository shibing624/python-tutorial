function ChkAllClick(sonName, cbAllId) {
    var arrSon = document.getElementsByName(sonName);
    var cbAll = document.getElementById(cbAllId);
    var tempState = cbAll.checked;
    for (i = 0; i < arrSon.length; i++) {
        if (arrSon[i].checked != tempState)
            arrSon[i].click();
    }
}
function ChkSonClick(sonName, cbAllId) {
    var arrSon = document.getElementsByName(sonName);
    var cbAll = document.getElementById(cbAllId);
    for (var i = 0; i < arrSon.length; i++) {
        if (!arrSon[i].checked) {
            cbAll.checked = false;
            return;
        }
    }
    cbAll.checked = true;
}
function ChkOppClick(sonName) {
    var arrSon = document.getElementsByName(sonName);
    for (i = 0; i < arrSon.length; i++) {
        arrSon[i].click();
    }
}
function changeBgColor(btn) {
    var btn = document.getElementById(btn);
    btn.style.backgroundColor = "#90BFFF"
}
function recoverBgColor(btn) {
    var btn = document.getElementById(btn);
    btn.style.backgroundColor = "#448EF3"
} 