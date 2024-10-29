const CHATBOT_FILL_HEIGHT = 100;
const SKETCHBOOK_FILL_HEIGHT = 30;

const old_content = {};

async function updateHeightsAndScroll() {
    const header = document.getElementById('header-title');
    const chatInput = document.getElementById('chat-input');
    const chatbot = document.getElementById('chatbot');
    const sideColumn = document.getElementById('side-column');
    const tabWrapper = document.getElementsByClassName('tab-wrapper')[0];
    const tabContents = document.getElementsByClassName('tab-content');

    if (header && chatInput && chatbot && sideColumn) {
        const headerHeight = header.offsetHeight;
        const chatInputHeight = chatInput.offsetHeight;

        const viewportHeight = window.innerHeight;

        const chatbotHeight = viewportHeight - CHATBOT_FILL_HEIGHT - headerHeight - chatInputHeight;

        chatbot.style.height = `${chatbotHeight}px`;
        sideColumn.style.height = `${chatbotHeight}px`;

        if (tabWrapper) {
            const contentHeight = chatbotHeight - SKETCHBOOK_FILL_HEIGHT - tabWrapper.offsetHeight;

            for (const content of tabContents) {
                content.style.height = `${contentHeight}px`;
            }
        }
    }

    for (const s of tabContents) {
        if (old_content[s.id] !== s.innerHTML) {
            s.scrollTop = s.scrollHeight;
            old_content[s.id] = s.innerHTML;
        }
    }
}

window.onload = updateHeightsAndScroll();

setInterval(updateHeightsAndScroll, 1);

window.addEventListener('resize', updateHeightsAndScroll);
