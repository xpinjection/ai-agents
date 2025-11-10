CREATE TABLE transactions
(
    transaction_id BIGSERIAL PRIMARY KEY,
    posted_at      TIMESTAMP      NOT NULL,
    merchant_name  TEXT,
    amount         NUMERIC(12, 2) NOT NULL,
    currency       CHAR(3)        NOT NULL,
    mcc            INTEGER        NOT NULL,
    card_last4     CHAR(4)        NOT NULL,
    account_id     BIGINT         NOT NULL
);