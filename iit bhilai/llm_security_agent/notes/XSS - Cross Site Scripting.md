

## refs 
https://owasp.org/www-community/attacks/DOM_Based_XSS
https://owasp.org/www-community/attacks/xss/
https://owasp.org/www-project-web-security-testing-guide/v41/4-Web_Application_Security_Testing/11-Client_Side_Testing/01-Testing_for_DOM-based_Cross_Site_Scripting.html
# Information
Malicious scripts are injected into otherwise benign and trusted websites .

XSS attacks occur when an attacker uses a web application to send malicious code, generally in the form of a browser side script, to a different end user.

Flaws that allow these attacks to succeed are quite widespread and occur anywhere a web application uses input from a user within the output it generates without validating or encoding it

Cross-Site Scripting (XSS) attacks occur when:

1. Data enters a Web application through an untrusted source, most frequently a web request.
2. The data is included in dynamic content that is sent to a web user without being validated for malicious content.

The malicious content sent to the web browser often takes the form of a segment of JavaScript, but may also include HTML, Flash, or any other type of code that the browser may execute. 

The variety of attacks based on XSS is almost limitless, but they commonly include transmitting private data, like cookies or other session information, to the attacker, redirecting the victim to web content controlled by the attacker, or performing other malicious operations on the user’s machine under the guise of the vulnerable site


categorized into two categories: reflected and stored.

## DOM Based XSS

DOM Based [XSS](https://owasp.org/www-community/attacks/XSS "wikilink") (or as it is called in some texts, “type-0 XSS”) is an XSS attack wherein the attack payload is executed as a result of ==modifying the DOM “environment” in the victim’s browser used by the original client side script==, so that the client side code runs in an “unexpected” manner. That is, the page itself (the HTTP response that is) does not change, but the client side code contained in the page executes differently due to the malicious modifications that have occurred in the DOM environment.

