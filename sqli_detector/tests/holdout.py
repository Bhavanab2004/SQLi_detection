"""Hand-written examples that are NOT used in training (neither the CSVs nor
augment.py templates). They approximate what real API clients send."""

BENIGN = [
    "alice", "bob.marley@example.com", "Rajesh Kumar", "42", "1001", "red running shoes size 9",
    "wireless headphones", "2025-03-14", "123 Main Street, Springfield", "Mary-Jane Watson",
    "I'd like a refund for order 5521", "O'Malley", "Select your preferred language", "winter jacket",
    "Where can I find the user manual?", "gift card $25", "p@ssw0rd!2024", "pending", "shipped",
    "Union Square Cafe", "How do I update my profile picture?", "hello world", "9876543210",
    "bengaluru", "price low to high", "sort by rating", "Table 5 - window seat", "ice cream (vanilla)",
    "john_doe99", "Delete this item from my cart", "cheap flights to goa", "100", "true", "en-US",
    "My order is late; please help!", "A+B=C", "50% discount", "it's 5 o'clock somewhere",
    "Drop-off location: Airport T2", "and/or", "review: 5 stars, loved it", "SKU-48213-B",
    "Insert your card to pay", "Sleep tracker app", "Order of the Phoenix", "from 9am to 5pm",
    "user@domain.co.in", "notebook", "latest news", "2",
]

MALICIOUS = [
    "' OR '1'='1", "' OR 1=1 --", "admin'--", "1 OR 1=1", "1' OR '1'='1' #",
    "' UNION SELECT username, password FROM users --", "1 UNION SELECT NULL, NULL, NULL",
    "1; DROP TABLE users", "'; DROP TABLE accounts; --", "1' AND SLEEP(5) --",
    "1; WAITFOR DELAY '0:0:5'--", "' OR pg_sleep(5)--", "1 AND 1=1", "1' AND 1=2 UNION SELECT 1,2,3 --",
    "' OR ''='", "x' OR 'x'='x", "1) OR (1=1", "') OR ('1'='1",
    "1' ORDER BY 10 --", "1 AND (SELECT COUNT(*) FROM users) > 0",
    "' AND extractvalue(1, concat(0x7e, version())) --", "1 AND updatexml(1,concat(0x7e,database()),1)",
    "' UNION SELECT table_name, NULL FROM information_schema.tables --",
    "1; EXEC xp_cmdshell('whoami')--", "1' AND (SELECT SUBSTRING(password,1,1) FROM users LIMIT 1)='a",
    "admin' OR '1'='1'/*", "' HAVING 1=1 --", "1' AND ASCII(SUBSTRING((SELECT database()),1,1))>64 --",
    "%27%20OR%20%271%27%3D%271", "1'; SELECT * FROM users WHERE 't'='t",
    "' OR SLEEP(3) #", "1 OR BENCHMARK(1000000,MD5('a'))", "'; INSERT INTO users VALUES('h','h'); --",
    "1' UNION ALL SELECT @@version, NULL --", "' OR 1=1 LIMIT 1 --", "x'; UPDATE users SET role='admin' WHERE 'a'='a",
    "1 AND (SELECT 1 FROM (SELECT SLEEP(2))a)", "' or true --", "1' GROUP BY 1 HAVING 1=1 --",
    "' UNION SELECT NULL,sqlite_version() --", "1' AND 1=CAST((SELECT version()) AS int) --",
    "admin' AND '1'='1", "1; SHUTDOWN --", "' OR EXISTS(SELECT * FROM users) --",
    "-1' UNION SELECT 1,load_file('/etc/passwd'),3 --", "1' AND IF(1=1,SLEEP(4),0) --",
    "a' OR 'b' LIKE 'b", "') UNION SELECT credit_card FROM payments --", "1 OR 5=5",
]
