*** Settings ***
Library    Selenium2Library

*** Variables ***
${browser}    chrome
${url}    https://google.co.th/

*** Keywords ***
Go to Google
    Open Browser    ${url}    ${browser}
    Maximize Browser Window

    # Click Element //h3[@class="iblpc"]

*** Test Cases ***
Google Index
    Go to Google
    Input Text name=q Robot Framework
    Click Button name=btnK