// ボタンを押した際の処理を追加
var buttonNode = document.querySelector('#loading-button');
buttonNode.addEventListener('click', function(e){
    // ボタン押された時点でローディングを表示
    showLoding();
    // 本来はここでWEB APIを呼び出して結果をまつ
    fetch('/')
        .then(function(res){
            // 結果を得たらローディングを消す
            // ローディングが見えるように今回は60秒ずらしている
            setTimeout(hideLoading, 100000);
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