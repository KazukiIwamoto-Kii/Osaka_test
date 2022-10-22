// ボタンを押した際の処理を追加
var buttonNode = document.querySelector('#loading-button');
buttonNode.addEventListener('click', function(e){
    // ボタン押された時点でローディングを表示
    showLoding();
    // 本来はここでWEB APIを呼び出して結果をまつ
    fetch('/')
        .then(function(res){
            // 結果を得たらローディングを消す
            // ローディングが見えるように今回は1000秒ずらしている
            setTimeout(hideLoading, 1000000);
        });
}, false);

function showLoding(){
    var loadingNode = document.querySelector('#loading');
    var loadingMessage = document.querySelector('#loading-message');
    loadingNode.style.display = '';
    loadingMessage.style.display = '';
}

function hideLoading(){
    var loadingNode = document.querySelector('#loading');
    var loadingMessage = document.querySelector('#loading-message');
    loadingNode.style.display = 'none';
    loadingMessage.style.display = 'none';

}

function checkLimit(){
    var count = 0;
    var Check = document.getElementsByClassName("name");
    for (i = 0; i < Check.length; i++){
        Flag[i] = i;
        if(Check[i].checked){
            Flag[i] = "chk";
            count++;
        }
    }
    if (count >= 3){
        for (i = 0; i < Check.length; i++){
            if (Flag[i] == "chk"){
                Check[i].disabled = false;
            } else {
                Check[i].disabled = true;
            }
        }
    } else {
        for (i = 0; i < Check.length; i++){
            Check[i].disabled = false;
        }
    }
}