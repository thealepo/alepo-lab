const messages = [
    "Are you sure?",
    "Really Sure?",
    "Are you positive?",
    "Pookie please...",
    "Think about it!",
    "If you say no, I'll cry...",
    "I will be sad...",
    "I will be very sad...",
    "I will be very very sad...",
    "I will be very very very sad...",
    "Okay fine, I'll stop asking...",
    "Just kidding, say yes please!"
]
let messageIndex = 0;

function handleNoClick(){
    const noButton = document.querySelector('.no-button');
    const yesButton = document.querySelector('.yes-button');
    noButton.textContent = messages[messageIndex];
    messageIndex = (messageIndex + 1) % messages.length;
    const currentSize = parseFloat(window.getComputedStyle(yesButton).fontSize);
    yesButton.style.fontSize = `${currentSize * 1.5}px`;
}
function handleYesClick(){
    window.location.href = "yes_page.html";
}