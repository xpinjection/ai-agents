from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Iterable, List, Dict


class BookingStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class Passenger:
    first_name: str
    last_name: str
    date_of_birth: date


@dataclass(frozen=True)
class Flight:
    code: str
    departure: str
    destination: str
    departure_time: datetime
    ticket_price: Decimal
    capacity: int


@dataclass(frozen=True)
class FlightInfo:
    flight: Flight
    available_capacity: int


@dataclass
class Booking:
    id: str
    flight_code: str
    passengers: List[Passenger]
    status: BookingStatus
    booked_at: datetime
    total_price: Decimal


class FlightNotFoundError(Exception): pass


class BookingNotFoundError(Exception): pass


class OverCapacityError(Exception): pass


class InputValidationError(Exception): pass


class FlightBookingService:
    def __init__(self, flights: Iterable[Flight] = ()):
        self._flights: Dict[str, Flight] = {f.code: f for f in flights}
        self._bookings: Dict[str, Booking] = {}
        self._lock = threading.RLock()

    def list_flights(self, departure: str, destination: str) -> List[FlightInfo]:
        dep = self._norm_city(departure)
        dst = self._norm_city(destination)
        out: List[FlightInfo] = []
        for f in self._flights.values():
            if f.departure == dep and f.destination == dst:
                out.append(FlightInfo(flight=f, available_capacity=self._available_capacity(f.code)))
        return sorted(out, key=lambda fi: fi.flight.departure_time)

    def get_flight(self, code: str) -> FlightInfo:
        flight = self._flights.get(code)
        if not flight:
            raise FlightNotFoundError(f"Flight '{code}' not found")
        return FlightInfo(flight=flight, available_capacity=self._available_capacity(code))

    def book_tickets(self, flight_code: str, passengers: List[Passenger]) -> Booking:
        if not passengers:
            raise InputValidationError("Passenger list must not be empty")

        self._validate_passengers(passengers)

        with (self._lock):
            finfo = self.get_flight(flight_code)
            if len(passengers) > finfo.available_capacity:
                raise OverCapacityError(
                    f"Not enough seats: requested {len(passengers)}, available {finfo.available_capacity}"
                )

            total = (finfo.flight.ticket_price * Decimal(len(passengers))).quantize(Decimal("0.01"),
                                                                                    rounding=ROUND_HALF_UP)

            booking = Booking(
                id=str(uuid.uuid4()),
                flight_code=finfo.flight.code,
                passengers=passengers,
                status=BookingStatus.CONFIRMED,
                booked_at=datetime.now(timezone.utc),
                total_price=total,
            )
            self._bookings[booking.id] = booking
            return booking

    def find_booking(self, booking_id: str) -> Booking:
        booking = self._bookings.get(booking_id)
        if not booking:
            raise BookingNotFoundError(f"Booking '{booking_id}' not found")
        return booking

    def cancel_booking(self, booking_id: str) -> Booking:
        with self._lock:
            booking = self.find_booking(booking_id)
            if booking.status == BookingStatus.CANCELLED:
                return booking
            booking.status = BookingStatus.CANCELLED
            return booking

    def _available_capacity(self, flight_code: str) -> int:
        flight = self._flights.get(flight_code)
        if not flight:
            raise FlightNotFoundError(f"Flight '{flight_code}' not found")

        active_seats = sum(
            len(b.passengers)
            for b in self._bookings.values()
            if b.flight_code == flight_code and b.status == BookingStatus.CONFIRMED
        )
        return max(0, flight.capacity - active_seats)

    @staticmethod
    def _norm_city(name: str) -> str:
        return name.strip().upper()

    @staticmethod
    def _validate_passengers(passengers: List[Passenger]) -> None:
        today = date.today()
        for i, p in enumerate(passengers, start=1):
            if not p.first_name:
                raise InputValidationError(f"Passenger #{i}: first_name is empty")
            if not p.last_name:
                raise InputValidationError(f"Passenger #{i}: last_name is empty")
            if p.date_of_birth > today:
                raise InputValidationError(f"Passenger #{i}: date_of_birth is in the future")


def _seed_flights() -> List[Flight]:
    return [
        Flight(
            code="LO123", departure="WARSAW",
            destination="KRAKOW",
            departure_time=datetime(2026, 1, 5, 9, 30),
            ticket_price=Decimal("199.99"),
            capacity=5,
        ),
        Flight(
            code="LO456",
            departure="WARSAW",
            destination="GDANSK",
            departure_time=datetime(2026, 1, 5, 13, 15),
            ticket_price=Decimal("159.50"),
            capacity=3,
        ),
        Flight(
            code="LO789",
            departure="KRAKOW",
            destination="WARSAW",
            departure_time=datetime(2026, 1, 6, 18, 45),
            ticket_price=Decimal("189.00"),
            capacity=4,
        ),
    ]


def _demo():
    svc = FlightBookingService(flights=_seed_flights())

    print("== List flights Warsaw -> Krakow ==")
    for fi in svc.list_flights("Warsaw", "Krakow"):
        print(fi)

    print("\n== Get flight LO123 ==")
    print(svc.get_flight("LO123"))

    print("\n== Book 2 tickets on LO123 ==")
    booking = svc.book_tickets(
        "LO123",
        [
            Passenger(first_name="Alice", last_name="Nowak", date_of_birth=date(1990, 3, 10)),
            Passenger(first_name="Bob", last_name="Kowalski", date_of_birth=date(1988, 7, 22)),
        ],
    )
    print(booking)

    print("\n== Capacity after booking ==")
    print(svc.get_flight("LO123"))

    print("\n== Find booking by id ==")
    print(svc.find_booking(booking.id))

    print("\n== Cancel booking ==")
    svc.cancel_booking(booking.id)
    print(svc.find_booking(booking.id))

    print("\n== Capacity after cancel ==")
    print(svc.get_flight("LO123"))


if __name__ == "__main__":
    _demo()
